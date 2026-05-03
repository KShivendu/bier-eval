from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from beir.retrieval.evaluation import EvaluateRetrieval

from bier_benches.corpus import passage_text

if TYPE_CHECKING:
    from bier_benches.protocols import LexicalBackend, PassageEncoder, VectorBackend

logger = logging.getLogger(__name__)


def corpus_to_lists(corpus: dict[str, dict]) -> tuple[list[str], list[str]]:
    doc_ids = list(corpus.keys())
    texts = [passage_text(corpus[d]) for d in doc_ids]
    return doc_ids, texts


def dense_retrieve(
    corpus: dict[str, dict],
    queries: dict[str, str],
    encoder: PassageEncoder,
    backend: VectorBackend,
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    doc_ids, texts = corpus_to_lists(corpus)
    logger.info("Encoding %d passages (dim=%d)", len(texts), encoder.embedding_dim)
    doc_emb = encoder.encode_passages(texts, batch_size=batch_size)
    backend.build(doc_ids, doc_emb)

    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    logger.info("Encoding %d queries", len(qtexts))
    q_emb = encoder.encode_queries(qtexts, batch_size=batch_size)
    ranked = backend.search(q_emb, top_k=top_k)

    results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(qids, ranked):
        results[qid] = {did: float(score) for did, score in hits}
    return results


def lexical_retrieve(
    corpus: dict[str, dict],
    queries: dict[str, str],
    backend: LexicalBackend,
    *,
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    doc_ids, texts = corpus_to_lists(corpus)
    backend.build(doc_ids, texts)
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    ranked = backend.search(qtexts, top_k=top_k)
    return {qid: {did: float(s) for did, s in hits} for qid, hits in zip(qids, ranked)}


def evaluate_results(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    evaluator = EvaluateRetrieval()
    return evaluator.evaluate(qrels, results, k_values)
