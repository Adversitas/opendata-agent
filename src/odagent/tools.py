"""Tools exposed to the agents: a read-only SQL runner and a document search."""

from __future__ import annotations

import re
import sqlite3
import time

MAX_ROWS = 20
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I
)


class SQLError(Exception):
    pass


def clean_sql(sql: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S | re.I)
    if m:
        sql = m.group(1)
    # small models often echo the "SQL:" label used in the prompt's examples
    sql = re.sub(r"^\s*(sql|requête)\s*:\s*", "", sql.strip(), flags=re.I)
    sql = sql.strip().rstrip(";").strip()
    return sql


def run_sql(conn: sqlite3.Connection, sql: str, timeout_s: float = 2.0) -> list[tuple]:
    """Execute one SELECT statement with a row limit and a time limit."""
    sql = clean_sql(sql)
    if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql or FORBIDDEN.search(sql):
        raise SQLError("only a single SELECT statement is allowed")
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
    except sqlite3.Error as exc:
        raise SQLError(str(exc)) from exc
    finally:
        conn.set_progress_handler(None, 0)
    return rows


def format_rows(rows: list[tuple]) -> str:
    if not rows:
        return "(aucune ligne)"
    return "\n".join(" | ".join(str(v) for v in r) for r in rows)
