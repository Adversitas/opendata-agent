"""Answer scoring: numbers, names and free-text spans (SQuAD-style, French normalisation)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .questions import Question

ARTICLES = re.compile(r"\b(le|la|les|l|un|une|des|du|de|d|au|aux)\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def numbers_in(text: str) -> set[int]:
    """Integers in a French or English answer: '45 312', '45 312', '45,312', '45.312'."""
    text = re.sub(r"(?<=\d)[\s  .,](?=\d{3}\b)", "", str(text))
    return {int(n) for n in re.findall(r"\d+", text)}


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def is_correct(q: Question, answer: str) -> bool:
    if q.answer_type == "number":
        return int(q.answers[0]) in numbers_in(answer)
    if q.answer_type == "name":
        gold = normalize(q.answers[0])
        return gold in normalize(answer) or token_f1(answer, q.answers[0]) >= 0.6
    na = normalize(answer)
    return (
        any(normalize(g) and normalize(g) in na for g in q.answers)
        or max(token_f1(answer, g) for g in q.answers) >= 0.5
    )


def best_f1(q: Question, answer: str) -> float:
    return max(token_f1(answer, g) for g in q.answers)
