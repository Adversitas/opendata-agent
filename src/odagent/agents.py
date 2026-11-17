"""Three assistant architectures, run in batches so a local GPU model stays busy.

* ``rag-only``: always search the documents, then answer. The usual RAG chatbot.
* ``single-agent``: one LLM call chooses a tool and writes its input (a search query or a SQL
  statement) as JSON; the tool runs; a failed SQL statement is sent back once with the error
  message; a second call writes the answer from the observation.
* ``router+specialists``: a router call classifies the question (documents vs table); a SQL
  specialist (schema + examples, one self-correction round) or a RAG specialist handles it;
  the answer is written from what the specialist returned.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

from .data import TABLE_SCHEMA
from .llm import Backend, Generation
from .questions import Question
from .retrieval import HybridRetriever
from .tools import SQLError, format_rows, run_sql

ANSWER_SYSTEM = (
    "Tu es un assistant qui répond en français, en une phrase courte, uniquement à partir des "
    "informations fournies. Donne d'abord la réponse exacte (nombre ou nom), sans calcul. Si "
    "l'information n'y est pas, réponds « Je ne sais pas »."
)

SQL_EXAMPLES = """Exemples :
Q: Combien de musées compte la région Corse ?
SQL: SELECT COUNT(*) FROM musees WHERE region = 'Corse'
Q: Quel musée de Lyon a eu le plus d'entrées payantes ?
SQL: SELECT nom, entrees_payantes FROM musees WHERE ville = 'Lyon' ORDER BY entrees_payantes DESC LIMIT 1
Q: Total des entrées des musées de la Martinique ?
SQL: SELECT SUM(entrees_totales) FROM musees WHERE region = 'Martinique'"""  # noqa: E501

SQL_SYSTEM = (
    "Tu écris une seule requête SQLite (SELECT) pour répondre à la question à partir de cette "
    f"table :\n{TABLE_SCHEMA}\n{SQL_EXAMPLES}\nRéponds uniquement avec la requête SQL."
)

ROUTER_SYSTEM = (
    "Classe la question selon la source qui permet d'y répondre. « table » : statistiques de "
    "fréquentation 2018 des musées de France (nombre de musées, entrées payantes, gratuites ou "
    "totales par musée, ville, département ou région). « documents » : toute autre question de "
    "culture générale (histoire, sciences, géographie, sport, personnes...), répondue par des "
    "articles Wikipédia. Réponds par un seul mot : table ou documents."
)

TOOLS_SYSTEM = (
    "Tu disposes de deux outils.\n"
    "1. run_sql : interroge la table SQLite suivante (statistiques 2018 des musées de France) :\n"
    f"{TABLE_SCHEMA}\n{SQL_EXAMPLES}\n"
    "2. search_documents : recherche dans des articles Wikipédia en français (culture "
    "générale, histoire, sciences, sport...).\n"
    'Réponds uniquement par un objet JSON : {"tool": "run_sql", "input": "<requête SELECT>"} '
    'ou {"tool": "search_documents", "input": "<requête de recherche>"}.'
)


@dataclass
class Trace:
    qid: str
    route: str = ""
    tool_input: str = ""
    observation: str = ""
    sql_error: str = ""
    sql_retried: bool = False
    retrieved: list[str] = field(default_factory=list)
    answer: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, g: Generation) -> str:
        self.calls += 1
        self.prompt_tokens += g.prompt_tokens
        self.completion_tokens += g.completion_tokens
        self.seconds += g.seconds
        return g.text.strip()


def _msgs(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Toolbox:
    def __init__(self, conn: sqlite3.Connection, retriever: HybridRetriever, k: int = 3) -> None:
        self.conn = conn
        self.retriever = retriever
        self.k = k

    def search(self, queries: list[str]) -> list[list[int]]:
        return self.retriever.search_many(queries, k=self.k)

    def context(self, hits: list[int]) -> str:
        return "\n\n".join(f"[{i + 1}] {self.retriever.texts[h]}" for i, h in enumerate(hits))


def _sql_round(backend: Backend, box: Toolbox, items: list[tuple[Trace, Question, str]]) -> None:
    """Execute SQL for each (trace, question, sql); retry failures once with the error."""
    retry = []
    for tr, q, sql in items:
        tr.tool_input = sql
        try:
            tr.observation = format_rows(run_sql(box.conn, sql))
        except SQLError as exc:
            tr.sql_error = str(exc)
            retry.append((tr, q, sql))
    if not retry:
        return
    prompts = [
        _msgs(
            SQL_SYSTEM,
            f"Question : {q.text}\nRequête précédente : {sql}\nErreur : "
            f"{tr.sql_error}\nÉcris une requête corrigée.",
        )
        for tr, q, sql in retry
    ]
    for (tr, _, _), g in zip(retry, backend.generate(prompts, max_new_tokens=160), strict=True):
        tr.sql_retried = True
        tr.tool_input = tr.add(g)
        try:
            tr.observation = format_rows(run_sql(box.conn, tr.tool_input))
            tr.sql_error = ""
        except SQLError as exc:
            tr.sql_error = str(exc)
            tr.observation = f"Erreur SQL : {exc}"


def _answer(backend: Backend, items: list[tuple[Trace, Question]]) -> None:
    prompts = []
    for tr, q in items:
        source = "Résultat de la requête SQL" if tr.route == "table" else "Extraits"
        extra = f"\nRequête : {tr.tool_input}" if tr.route == "table" else ""
        prompts.append(
            _msgs(ANSWER_SYSTEM, f"{source} :{extra}\n{tr.observation}\n\nQuestion : {q.text}")
        )
    for (tr, _), g in zip(items, backend.generate(prompts, max_new_tokens=80), strict=True):
        tr.answer = tr.add(g)


def _rag(box: Toolbox, items: list[tuple[Trace, Question, str]]) -> None:
    hits = box.search([query for _, _, query in items])
    for (tr, _, query), h in zip(items, hits, strict=True):
        tr.route, tr.tool_input = "documents", query
        tr.retrieved = [box.retriever.doc_ids[i] for i in h]
        tr.observation = box.context(h)


def rag_only(backend: Backend, box: Toolbox, questions: list[Question]) -> list[Trace]:
    traces = [Trace(q.qid) for q in questions]
    _rag(box, [(t, q, q.text) for t, q in zip(traces, questions, strict=True)])
    _answer(backend, list(zip(traces, questions, strict=True)))
    return traces


def parse_tool_call(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            tool = str(obj.get("tool", "")).strip()
            if tool in {"run_sql", "search_documents"}:
                return tool, str(obj.get("input", "")).strip()
        except json.JSONDecodeError:
            pass
    if re.search(r"\bselect\b", text, re.I):
        return "run_sql", text
    return "search_documents", ""


def single_agent(backend: Backend, box: Toolbox, questions: list[Question]) -> list[Trace]:
    traces = [Trace(q.qid) for q in questions]
    prompts = [_msgs(TOOLS_SYSTEM, f"Question : {q.text}") for q in questions]
    sql_items, rag_items = [], []
    for tr, q, g in zip(traces, questions, backend.generate(prompts, 200), strict=True):
        tool, tool_input = parse_tool_call(tr.add(g))
        if tool == "run_sql":
            tr.route = "table"
            sql_items.append((tr, q, tool_input))
        else:
            rag_items.append((tr, q, tool_input or q.text))
    _sql_round(backend, box, sql_items)
    if rag_items:
        _rag(box, rag_items)
    _answer(backend, list(zip(traces, questions, strict=True)))
    return traces


def router_specialists(backend: Backend, box: Toolbox, questions: list[Question]) -> list[Trace]:
    traces = [Trace(q.qid) for q in questions]
    routes = backend.generate([_msgs(ROUTER_SYSTEM, q.text) for q in questions], 4)
    table, docs = [], []
    for tr, q, g in zip(traces, questions, routes, strict=True):
        label = tr.add(g).lower()
        (table if label.startswith("table") else docs).append((tr, q))
    if table:
        sqls = backend.generate([_msgs(SQL_SYSTEM, f"Question : {q.text}") for _, q in table], 160)
        items = []
        for (tr, q), g in zip(table, sqls, strict=True):
            tr.route = "table"
            items.append((tr, q, tr.add(g)))
        _sql_round(backend, box, items)
    if docs:
        _rag(box, [(tr, q, q.text) for tr, q in docs])
    _answer(backend, list(zip(traces, questions, strict=True)))
    return traces


AGENTS = {
    "rag-only": rag_only,
    "single-agent": single_agent,
    "router+specialists": router_specialists,
}
