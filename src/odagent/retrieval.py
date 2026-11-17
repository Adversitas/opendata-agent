"""Hybrid retrieval over the PIAF paragraphs: BM25 + dense embeddings fused with RRF."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

_STOP_WORDS = """
le la les l un une des de du d et en à a au aux pour par sur dans est sont qui que quoi quel
quelle quels quelles ce cette ces se sa son ses leur leurs il elle ils elles on ne pas plus avec
comme été être t combien quand où comment pourquoi y
"""


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


STOP = frozenset(_strip_accents(_STOP_WORDS).split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", _strip_accents(text))
    return [t for t in words if t not in STOP and len(t) > 1]


class BM25Index:
    def __init__(self, docs: list[str]) -> None:
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))


class DenseIndex:
    """multilingual-e5 style encoder ("query: " / "passage: " prefixes)."""

    def __init__(self, docs: list[str], model_id: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)
        self.emb = self.model.encode(
            [f"passage: {d}" for d in docs], normalize_embeddings=True, batch_size=64
        )

    def scores_many(self, queries: list[str]) -> np.ndarray:
        q = self.model.encode([f"query: {x}" for x in queries], normalize_embeddings=True)
        return q @ self.emb.T


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of several score vectors (higher score = better)."""
    fused = np.zeros(len(rankings[0]))
    for scores in rankings:
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class HybridRetriever:
    def __init__(self, doc_ids: list[str], texts: list[str], dense: DenseIndex | None = None):
        self.doc_ids = doc_ids
        self.texts = texts
        self.bm25 = BM25Index(texts)
        self.dense = dense

    def search_many(self, queries: list[str], k: int = 3, mode: str = "hybrid") -> list[list[int]]:
        dense_scores = (
            self.dense.scores_many(queries) if self.dense is not None and mode != "bm25" else None
        )
        out = []
        for i, q in enumerate(queries):
            if mode == "bm25" or dense_scores is None:
                s = self.bm25.scores(q)
            elif mode == "dense":
                s = dense_scores[i]
            else:
                s = rrf([self.bm25.scores(q), dense_scores[i]])
            out.append(list(np.argsort(-s)[:k]))
        return out
