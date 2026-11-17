"""Evaluation questions with gold answers.

* Table questions are generated from templates; their gold answers are computed with pandas
  (independently of any SQL the agents write).
* Document questions are a seeded sample of PIAF questions with the volunteers' answers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

METROPOLE = [
    "Auvergne-Rhône-Alpes",
    "Bourgogne-Franche-Comté",
    "Bretagne",
    "Centre-Val de Loire",
    "Grand Est",
    "Hauts-de-France",
    "Île-de-France",
    "Normandie",
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Pays de la Loire",
    "Provence-Alpes-Côte d'Azur",
]


@dataclass
class Question:
    qid: str
    source: str  # "table" | "documents"
    kind: str
    text: str
    answers: list[str]
    answer_type: str  # "number" | "name" | "span"
    doc_id: str | None = None
    meta: dict = field(default_factory=dict)


def table_questions(museums: pd.DataFrame, n: int = 60, seed: int = 0) -> list[Question]:
    rng = random.Random(seed)
    cities = museums.groupby("ville").size()
    cities = sorted(cities[cities >= 5].index)
    m = museums
    templates = []

    def count_region(r):
        return (
            f"Combien de musées de France la région {r} compte-t-elle dans les données "
            "de fréquentation 2018 ?",
            [str(int((m.region == r).sum()))],
            "number",
        )

    def total_region(r):
        return (
            f"Quel est le nombre total d'entrées (payantes et gratuites) des musées de la région "
            f"{r} en 2018 ?",
            [str(int(m.loc[m.region == r, "entrees_totales"].sum()))],
            "number",
        )

    def top_region(r):
        sub = m[m.region == r].sort_values("entrees_totales", ascending=False)
        return (
            f"Quel musée de la région {r} a reçu le plus d'entrées en 2018 ?",
            [sub.iloc[0]["nom"]],
            "name",
        )

    def over_region(r):
        threshold = rng.choice([10000, 50000, 100000])
        return (
            f"Combien de musées de la région {r} ont dépassé {threshold} entrées au total en "
            "2018 ?",
            [str(int(((m.region == r) & (m.entrees_totales > threshold)).sum()))],
            "number",
        )

    def total_city(c):
        return (
            f"Combien d'entrées au total les musées de {c} ont-ils enregistrées en 2018 ?",
            [str(int(m.loc[m.ville == c, "entrees_totales"].sum()))],
            "number",
        )

    def free_city(c):
        return (
            f"Combien d'entrées gratuites les musées de {c} ont-ils comptabilisées en 2018 ?",
            [str(int(m.loc[m.ville == c, "entrees_gratuites"].sum()))],
            "number",
        )

    for fn in (count_region, total_region, top_region, over_region):
        templates += [(fn.__name__, fn, r) for r in METROPOLE]
    for fn in (total_city, free_city):
        templates += [(fn.__name__, fn, c) for c in cities]
    rng.shuffle(templates)
    out = []
    for i, (kind, fn, arg) in enumerate(templates[:n]):
        text, answers, atype = fn(arg)
        out.append(Question(f"t{i:03d}", "table", kind, text, answers, atype, meta={"arg": arg}))
    return out


def document_questions(
    piaf: pd.DataFrame, corpus: pd.DataFrame, n: int = 60, seed: int = 0
) -> list[Question]:
    ids = dict(zip(corpus["context"], corpus["doc_id"], strict=True))
    sample = piaf.sample(n=min(n, len(piaf)), random_state=seed)
    return [
        Question(
            f"d{i:03d}",
            "documents",
            "piaf",
            row.question.strip(),
            list(row.answer),
            "span",
            doc_id=ids[row.context],
        )
        for i, row in enumerate(sample.itertuples())
    ]
