import json

import pandas as pd
import pytest

from odagent import data
from odagent.agents import AGENTS, Toolbox, parse_tool_call
from odagent.llm import FakeBackend
from odagent.questions import Question, document_questions, table_questions
from odagent.retrieval import HybridRetriever, rrf, tokenize
from odagent.scoring import is_correct, normalize, numbers_in, token_f1
from odagent.tools import SQLError, clean_sql, run_sql


@pytest.fixture(scope="module")
def museums() -> pd.DataFrame:
    return data.load_museums()


@pytest.fixture(scope="module")
def conn(museums):
    return data.connect(museums)


def test_museums_table(museums):
    assert len(museums) > 1000
    assert (museums.entrees_totales == museums.entrees_payantes + museums.entrees_gratuites).all()
    assert museums.loc[museums.ville == "Lyon", "region"].eq("Auvergne-Rhône-Alpes").all()
    assert (museums.region == "Inconnue").sum() <= 2


def test_table_questions_gold_matches_sql(museums, conn):
    qs = table_questions(museums, n=200)
    assert len({q.kind for q in qs}) == 6
    for q in qs:
        if q.kind == "count_region":
            region = q.meta["arg"].replace("'", "''")
            (n,) = run_sql(conn, f"SELECT COUNT(*) FROM musees WHERE region = '{region}'")[0]
            assert str(n) == q.answers[0]
        if q.kind == "total_city":
            city = q.meta["arg"].replace("'", "''")
            (n,) = run_sql(conn, f"SELECT SUM(entrees_totales) FROM musees WHERE ville='{city}'")[0]
            assert str(n) == q.answers[0]


def test_document_questions_reference_corpus():
    piaf = data.load_piaf()
    corpus = data.piaf_corpus(piaf)
    qs = document_questions(piaf, corpus, n=20)
    assert len(qs) == 20 and all(q.doc_id in set(corpus.doc_id) for q in qs)


def test_sql_guardrails(conn):
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert clean_sql("SQL: SELECT 1") == "SELECT 1"
    assert run_sql(conn, "SELECT COUNT(*) FROM musees")[0][0] > 1000
    for bad in ["DROP TABLE musees", "SELECT 1; DELETE FROM musees", "PRAGMA table_info(m)"]:
        with pytest.raises(SQLError):
            run_sql(conn, bad)
    with pytest.raises(SQLError):
        run_sql(conn, "SELECT nope FROM musees")
    assert len(run_sql(conn, "SELECT * FROM musees")) == 20


def test_scoring():
    assert numbers_in("Il y a 45 312 entrées, soit 45 312.") == {45312}
    assert numbers_in("1,234,567 visitors") == {1234567}
    assert normalize("L'Église de la Madeleine") == "eglise madeleine"
    assert token_f1("le musée du Louvre", "Louvre") == pytest.approx(2 / 3)
    num = Question("t", "table", "k", "?", ["37"], "number")
    assert is_correct(num, "La région compte 37 musées.") and not is_correct(num, "38")
    span = Question("d", "documents", "piaf", "?", ["100 000"], "span")
    assert is_correct(span, "Environ 100 000 personnes.")


def test_retrieval_bm25_and_rrf():
    texts = ["Le Louvre est un musée à Paris.", "La Loire est un fleuve.", "Lyon est en France."]
    r = HybridRetriever(["a", "b", "c"], texts)
    assert r.search_many(["Quel fleuve ?"], k=1)[0] == [1]
    assert tokenize("Où est l'Église ?") == ["eglise"]
    fused = rrf([pd.Series([3, 2, 1]).to_numpy(), pd.Series([1, 2, 3]).to_numpy()])
    assert fused[0] == pytest.approx(fused[2])


def test_parse_tool_call():
    assert parse_tool_call('{"tool": "run_sql", "input": "SELECT 1"}') == ("run_sql", "SELECT 1")
    assert parse_tool_call("SELECT COUNT(*) FROM musees")[0] == "run_sql"
    assert parse_tool_call("je ne sais pas") == ("search_documents", "")


def _scripted(museums):
    """A fake LLM that behaves like a competent model, to exercise every agent path."""

    def fn(msgs):
        system, user = msgs[0]["content"], msgs[-1]["content"]
        if system.startswith("Classe"):
            return "table" if "musée" in user else "documents"
        if system.startswith("Tu disposes"):
            if "musée" in user:
                return json.dumps({"tool": "run_sql", "input": "SELECT COUNT(*) FROM musee"})
            return json.dumps({"tool": "search_documents", "input": "fleuve"})
        if system.startswith("Tu écris"):
            if "Erreur" in user:
                return "SELECT COUNT(*) FROM musees WHERE region = 'Bretagne'"
            return "SELECT COUNT(*) FROM musee WHERE region = 'Bretagne'"  # typo on purpose
        if "SQL" in user:
            return "Résultat : " + user.split(":\n", 1)[-1].split("\n")[1]
        return "La Loire"

    return FakeBackend(fn)


def test_agents_end_to_end(museums, conn):
    texts = ["La Loire est le plus long fleuve de France.", "Le Louvre est à Paris."]
    box = Toolbox(conn, HybridRetriever(["p0", "p1"], texts), k=1)
    gold = str(int((museums.region == "Bretagne").sum()))
    qs = [
        Question("t", "table", "count_region", "Combien de musée en Bretagne ?", [gold], "number"),
        Question(
            "d",
            "documents",
            "piaf",
            "Quel est le plus long fleuve ?",
            ["Loire"],
            "span",
            doc_id="p0",
        ),
    ]
    backend = _scripted(museums)
    router = AGENTS["router+specialists"](backend, box, qs)
    assert [t.route for t in router] == ["table", "documents"]
    assert router[0].sql_retried and not router[0].sql_error
    assert is_correct(qs[0], router[0].answer) and is_correct(qs[1], router[1].answer)
    assert router[1].retrieved == ["p0"]

    single = AGENTS["single-agent"](backend, box, qs)
    assert single[0].route == "table" and single[0].sql_retried
    rag = AGENTS["rag-only"](backend, box, qs)
    assert {t.route for t in rag} == {"documents"}
