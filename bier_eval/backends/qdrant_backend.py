from __future__ import annotations

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class QdrantVectorBackend:
    """Remote or local Qdrant. Uses DOT distance — pair with L2-normalized vectors for cosine-like ranking."""

    def __init__(
        self,
        collection: str,
        *,
        url: str | None = "http://localhost:6333",
        api_key: str | None = None,
        grpc_port: int | None = None,
        recreate_collection: bool = True,
    ) -> None:
        from qdrant_client import QdrantClient

        kwargs: dict = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        if grpc_port is not None:
            kwargs["grpc_port"] = grpc_port
        self._client = QdrantClient(**kwargs)
        self.collection = collection
        self.recreate_collection = recreate_collection

    def build(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        v = np.asarray(vectors, dtype=np.float32)
        dim = v.shape[1]
        if self.recreate_collection and self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.DOT),
            )
        batch = 256
        for start in range(0, len(doc_ids), batch):
            end = min(start + batch, len(doc_ids))
            points = [
                PointStruct(id=_point_id(doc_ids[i]), vector=v[i].tolist(), payload={"doc_id": doc_ids[i]})
                for i in range(start, end)
            ]
            self._client.upsert(collection_name=self.collection, points=points)
        logger.info("Qdrant upserted %d points into %r", len(doc_ids), self.collection)

    def search(self, query_vectors: np.ndarray, *, top_k: int) -> list[list[tuple[str, float]]]:
        q = np.asarray(query_vectors, dtype=np.float32)
        out: list[list[tuple[str, float]]] = []
        for row in q:
            res = self._client.search(
                collection_name=self.collection,
                query_vector=row.tolist(),
                limit=top_k,
                with_payload=True,
            )
            hits: list[tuple[str, float]] = []
            for hit in res:
                did = hit.payload.get("doc_id") if hit.payload else None
                if did is None:
                    continue
                hits.append((str(did), float(hit.score)))
            out.append(hits)
        return out
