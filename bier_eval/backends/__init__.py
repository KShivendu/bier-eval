from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ElasticsearchVectorBackend",
    "FaissFlatIPBackend",
    "QdrantVectorBackend",
    "TantivyLexicalBackend",
]

if TYPE_CHECKING:
    from bier_eval.backends.elasticsearch_backend import ElasticsearchVectorBackend
    from bier_eval.backends.faiss_backend import FaissFlatIPBackend
    from bier_eval.backends.qdrant_backend import QdrantVectorBackend
    from bier_eval.backends.tantivy_backend import TantivyLexicalBackend


def __getattr__(name: str):
    if name == "FaissFlatIPBackend":
        from bier_eval.backends.faiss_backend import FaissFlatIPBackend

        return FaissFlatIPBackend
    if name == "QdrantVectorBackend":
        from bier_eval.backends.qdrant_backend import QdrantVectorBackend

        return QdrantVectorBackend
    if name == "ElasticsearchVectorBackend":
        from bier_eval.backends.elasticsearch_backend import ElasticsearchVectorBackend

        return ElasticsearchVectorBackend
    if name == "TantivyLexicalBackend":
        from bier_eval.backends.tantivy_backend import TantivyLexicalBackend

        return TantivyLexicalBackend
    raise AttributeError(name)
