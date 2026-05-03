from __future__ import annotations

import numpy as np


class SentenceTransformerEncoder:
    """HuggingFace SentenceTransformer wrapper; L2-normalizes outputs for dot-product search."""

    def __init__(self, model_name: str, *, normalize: bool = True, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 10_000,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        return self.encode_passages(texts, batch_size=batch_size)


class CallableEncoder:
    """Wrap any (texts) -> ndarray[float32] function for passages and queries."""

    def __init__(self, dim: int, encode_fn, *, encode_queries_fn=None) -> None:
        self.embedding_dim = dim
        self._encode = encode_fn
        self._encode_queries = encode_queries_fn or encode_fn

    def encode_passages(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size
        out = self._encode(texts)
        return np.asarray(out, dtype=np.float32)

    def encode_queries(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size
        out = self._encode_queries(texts)
        return np.asarray(out, dtype=np.float32)
