from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ElasticsearchVectorBackend:
    """Elasticsearch 8+ with `dense_vector` + kNN (dot_product similarity)."""

    def __init__(
        self,
        index: str,
        *,
        hosts: list[str] | str = "http://localhost:9200",
        basic_auth: tuple[str, str] | None = None,
        api_key: str | tuple[str, str] | None = None,
        delete_if_exists: bool = True,
        num_candidates: int | None = None,
    ) -> None:
        from elasticsearch import Elasticsearch

        es_kwargs: dict = {"hosts": hosts}
        if basic_auth:
            es_kwargs["basic_auth"] = basic_auth
        if api_key:
            es_kwargs["api_key"] = api_key
        self._es = Elasticsearch(**es_kwargs)
        self.index = index
        self.delete_if_exists = delete_if_exists
        self.num_candidates = num_candidates

    def build(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        dim = int(v.shape[1])
        if self._es.indices.exists(index=self.index):
            if self.delete_if_exists:
                self._es.indices.delete(index=self.index)
            else:
                raise RuntimeError(f"Elasticsearch index {self.index!r} already exists")

        self._es.indices.create(
            index=self.index,
            mappings={
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "dot_product",
                    },
                }
            },
        )

        from elasticsearch.helpers import bulk

        def gen():
            for i, doc_id in enumerate(doc_ids):
                yield {
                    "_index": self.index,
                    "_id": doc_id,
                    "_source": {"doc_id": doc_id, "embedding": v[i].tolist()},
                }

        bulk(self._es, gen(), refresh="wait_for", raise_on_error=True)
        logger.info("Elasticsearch indexed %d docs into %r", len(doc_ids), self.index)

    def search(self, query_vectors: np.ndarray, *, top_k: int) -> list[list[tuple[str, float]]]:
        q = np.asarray(query_vectors, dtype=np.float32)
        nc = self.num_candidates if self.num_candidates is not None else max(top_k * 4, 100)
        out: list[list[tuple[str, float]]] = []
        for row in q:
            body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": row.tolist(),
                    "k": top_k,
                    "num_candidates": min(nc, 10_000),
                },
                "_source": ["doc_id"],
                "size": top_k,
            }
            resp = self._es.search(index=self.index, body=body)
            hits: list[tuple[str, float]] = []
            for h in resp.get("hits", {}).get("hits", []):
                src = h.get("_source") or {}
                did = src.get("doc_id")
                if did is None:
                    did = h.get("_id")
                score = float(h.get("_score") or 0.0)
                hits.append((str(did), score))
            out.append(hits)
        return out
