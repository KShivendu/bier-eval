from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PassageEncoder(Protocol):
    """Maps query/passage strings to L2-normalized float32 vectors (recommended for dot-product backends)."""

    embedding_dim: int

    def encode_passages(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray: ...

    def encode_queries(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray: ...


@runtime_checkable
class VectorBackend(Protocol):
    """Dense vector index: build once, then batch-search query embeddings."""

    def build(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        """vectors shape (n_docs, dim), float32."""

    def search(self, query_vectors: np.ndarray, *, top_k: int) -> list[list[tuple[str, float]]]:
        """query_vectors shape (n_queries, dim). Returns parallel lists of (doc_id, score)."""


@runtime_checkable
class LexicalBackend(Protocol):
    """Lexical / sparse retrieval over passage text (BM25-style scoring from the engine)."""

    def build(self, doc_ids: list[str], texts: list[str]) -> None: ...

    def search(self, queries: list[str], *, top_k: int) -> list[list[tuple[str, float]]]:
        """Parallel to queries."""
