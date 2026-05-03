from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FaissFlatIPBackend:
    """In-process exact inner-product search (use L2-normalized vectors for cosine)."""

    def __init__(self, *, normalize: bool = True) -> None:
        import faiss

        self._faiss = faiss
        self.normalize = normalize
        self._index = None
        self._doc_ids: list[str] = []

    def build(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = np.asarray(vectors, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms
        dim = vectors.shape[1]
        self._doc_ids = list(doc_ids)
        index = self._faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        logger.info("FAISS IndexFlatIP: %d vectors, dim=%d", len(self._doc_ids), dim)

    def search(self, query_vectors: np.ndarray, *, top_k: int) -> list[list[tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("build() first")
        q = np.asarray(query_vectors, dtype=np.float32)
        if self.normalize:
            n = np.linalg.norm(q, axis=1, keepdims=True)
            n = np.maximum(n, 1e-12)
            q = q / n
        scores, idx = self._index.search(q, min(top_k, len(self._doc_ids)))
        out: list[list[tuple[str, float]]] = []
        for row_scores, row_idx in zip(scores, idx):
            hits: list[tuple[str, float]] = []
            for s, i in zip(row_scores, row_idx):
                if i < 0:
                    continue
                hits.append((self._doc_ids[int(i)], float(s)))
            out.append(hits)
        return out
