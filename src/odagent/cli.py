"""Command line: ``odagent retrieval | eval | report | ask``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .agents import AGENTS, Toolbox
from .questions import document_questions, table_questions
from .retrieval import DenseIndex, HybridRetriever
from .scoring import best_f1, is_correct


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _retriever(dense: bool = True) -> tuple[HybridRetriever, pd.DataFrame]:
    corpus = data.piaf_corpus(data.load_piaf())
    texts = corpus["context"].tolist()
    return HybridRetriever(
        corpus["doc_id"].tolist(), texts, DenseIndex(texts) if dense else None
    ), corpus


def cmd_retrieval(args: argparse.Namespace) -> int:
    piaf = data.load_piaf()
    retr, corpus = _retriever(dense=not args.bm25_only)
    ids = dict(zip(corpus["context"], corpus["doc_id"], strict=True))
    gold = [ids[c] for c in piaf["context"]]
    rows = []
    for mode in ["bm25"] if args.bm25_only else ["bm25", "dense", "hybrid"]:
        hits = retr.search_many(piaf["question"].tolist(), k=10, mode=mode)
        found = [[retr.doc_ids[h] for h in hs] for hs in hits]
        row = {"retriever": mode, "questions": len(gold)}
        for k in (1, 3, 5, 10):
            row[f"recall@{k}"] = float(
                np.mean([g in f[:k] for g, f in zip(gold, found, strict=True)])
            )
        ranks = [f.index(g) + 1 if g in f else None for g, f in zip(gold, found, strict=True)]
        row["mrr@10"] = float(np.mean([1 / r if r else 0 for r in ranks]))
        rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "retrieval.csv", index=False)
    print(table.to_string(index=False))
    return 0


def _backend(model: str, batch_size: int):
    from .llm import HFBackend

    return HFBackend(model, batch_size=batch_size)


def _questions(args) -> list:
    museums = data.load_museums()
    piaf = data.load_piaf()
    corpus = data.piaf_corpus(piaf)
    return table_questions(museums, args.n_table, args.seed) + document_questions(
        piaf, corpus, args.n_docs, args.seed
    )


def summarize(questions, traces_by_agent: dict) -> pd.DataFrame:
    rows = []
    for agent, traces in traces_by_agent.items():
        for q, tr in zip(questions, traces, strict=True):
            rows.append(
                {
                    "agent": agent,
                    "qid": q.qid,
                    "source": q.source,
                    "kind": q.kind,
                    "route_ok": tr.route == q.source,
                    "correct": is_correct(q, tr.answer),
                    "f1": best_f1(q, tr.answer),
                    "retrieval_hit": (q.doc_id in tr.retrieved) if q.doc_id else None,
                    "sql_failed": bool(tr.sql_error),
                    "sql_retried": tr.sql_retried,
                    "calls": tr.calls,
                    "tokens": tr.prompt_tokens + tr.completion_tokens,
                    "seconds": tr.seconds,
                }
            )
    return pd.DataFrame(rows)


def cmd_eval(args: argparse.Namespace) -> int:
    questions = _questions(args)
    retr, _ = _retriever()
    box = Toolbox(data.connect(data.load_museums()), retr, k=args.k)
    backend = _backend(args.model, args.batch_size)
    out = Path(args.out) / args.model.split("/")[-1].lower()
    out.mkdir(parents=True, exist_ok=True)
    traces = {}
    for name in args.agents:
        traces[name] = AGENTS[name](backend, box, questions)
        with (out / f"traces_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for q, tr in zip(questions, traces[name], strict=True):
                fh.write(
                    json.dumps({"question": asdict(q), "trace": asdict(tr)}, ensure_ascii=False)
                    + "\n"
                )
    per_q = summarize(questions, traces)
    per_q.insert(0, "model", out.name)
    previous = out / "per_question.csv"
    if previous.exists():  # keep agents that were not re-run this time
        old = pd.read_csv(previous)
        per_q = pd.concat([old[~old["agent"].isin(args.agents)], per_q], ignore_index=True)
    per_q.to_csv(previous, index=False)
    print(per_q.groupby(["agent", "source"])["correct"].mean().unstack().round(3))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.reports)
    per_q = pd.concat([pd.read_csv(p) for p in sorted(root.glob("*/per_question.csv"))])
    g = per_q.groupby(["model", "agent"])
    overall = g.agg(
        accuracy=("correct", "mean"),
        routing_accuracy=("route_ok", "mean"),
        llm_calls_per_q=("calls", "mean"),
        tokens_per_q=("tokens", "mean"),
        seconds_per_q=("seconds", "mean"),
    ).reset_index()
    by_src = per_q.pivot_table(
        index=["model", "agent"], columns="source", values="correct", aggfunc="mean"
    ).reset_index()
    by_src.columns = ["model", "agent"] + [f"accuracy_{c}" for c in by_src.columns[2:]]
    table = overall.merge(by_src, on=["model", "agent"])
    tq = (
        per_q[per_q.source == "table"]
        .groupby(["model", "agent"])
        .agg(sql_failed=("sql_failed", "mean"), sql_retried=("sql_retried", "mean"))
        .reset_index()
    )
    dq = (
        per_q[per_q.source == "documents"]
        .groupby(["model", "agent"])
        .agg(doc_f1=("f1", "mean"), retrieval_hit=("retrieval_hit", "mean"))
        .reset_index()
    )
    kinds = per_q.pivot_table(
        index=["model", "agent"], columns="kind", values="correct", aggfunc="mean"
    ).reset_index()
    lines = [
        "# Results",
        "",
        "## End to end",
        "",
        _md(
            table[
                [
                    "model",
                    "agent",
                    "accuracy",
                    "accuracy_table",
                    "accuracy_documents",
                    "routing_accuracy",
                    "llm_calls_per_q",
                    "tokens_per_q",
                    "seconds_per_q",
                ]
            ]
        ),
        "",
        "## Table questions: SQL execution",
        "",
        _md(tq),
        "",
        "## Document questions: retrieval and answer overlap",
        "",
        _md(dq),
        "",
        "## Accuracy by question template",
        "",
        _md(kinds),
    ]
    if (root / "retrieval.csv").exists():
        lines += [
            "",
            "## Retrieval on all 3,835 PIAF questions (761 paragraphs)",
            "",
            _md(pd.read_csv(root / "retrieval.csv")),
        ]
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "results.md").read_text(encoding="utf-8"))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .questions import Question

    retr, _ = _retriever()
    box = Toolbox(data.connect(data.load_museums()), retr)
    q = Question("q", "?", "adhoc", args.question, [""], "span")
    tr = AGENTS[args.agent](_backend(args.model, 1), box, [q])[0]
    print(json.dumps(asdict(tr), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odagent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval", help="recall@k of BM25 / dense / hybrid on PIAF")
    r.add_argument("--out", default="reports")
    r.add_argument("--bm25-only", action="store_true")
    r.set_defaults(func=cmd_retrieval)

    e = sub.add_parser("eval", help="run the assistant architectures on the question set")
    e.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    e.add_argument("--agents", nargs="*", default=list(AGENTS))
    e.add_argument("--n-table", type=int, default=60)
    e.add_argument("--n-docs", type=int, default=60)
    e.add_argument("--k", type=int, default=3)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--out", default="reports")
    e.set_defaults(func=cmd_eval)

    rep = sub.add_parser("report", help="build reports/results.md")
    rep.add_argument("--reports", default="reports")
    rep.set_defaults(func=cmd_report)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--agent", default="router+specialists", choices=list(AGENTS))
    a.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    a.set_defaults(func=cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
