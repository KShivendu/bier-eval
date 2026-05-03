from __future__ import annotations

import argparse
import logging
import sys

from beir import util

from bier_eval.backends import (
    ElasticsearchVectorBackend,
    FaissFlatIPBackend,
    QdrantVectorBackend,
    TantivyLexicalBackend,
)
from bier_eval.beir_eval import dense_retrieve, evaluate_results, lexical_retrieve
from bier_eval.encoders import SentenceTransformerEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BEIR retrieval with pluggable vector / lexical backends.")
    p.add_argument("--dataset", default="scifact", help="BEIR dataset name (e.g. scifact, nfcorpus).")
    p.add_argument("--data-dir", default="datasets", help="Root folder for downloaded BEIR zips.")
    p.add_argument(
        "--backend",
        choices=("faiss", "qdrant", "elasticsearch", "tantivy"),
        default="faiss",
    )
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer id (dense backends).")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k-values", default="1,3,5,10,100", help="Comma-separated cutoffs for BEIR metrics.")

    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--qdrant-collection", default="beir_dense")
    p.add_argument("--qdrant-api-key", default=None)
    p.add_argument("--no-qdrant-recreate", action="store_true")

    p.add_argument("--es-hosts", default="http://localhost:9200", help="Comma-separated Elasticsearch URLs.")
    p.add_argument("--es-index", default="beir_dense")
    p.add_argument("--no-es-delete", action="store_true")

    args = p.parse_args(argv)

    zip_url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    )
    data_folder = util.download_and_unzip(zip_url, args.data_dir)
    from beir.datasets.data_loader import GenericDataLoader

    corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split="test")
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    if args.backend == "tantivy":
        backend = TantivyLexicalBackend()
        results = lexical_retrieve(corpus, queries, backend, top_k=args.top_k)
    elif args.backend == "faiss":
        encoder = SentenceTransformerEncoder(args.model)
        backend = FaissFlatIPBackend()
        results = dense_retrieve(
            corpus,
            queries,
            encoder,
            backend,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
    elif args.backend == "qdrant":
        encoder = SentenceTransformerEncoder(args.model)
        backend = QdrantVectorBackend(
            args.qdrant_collection,
            url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            recreate_collection=not args.no_qdrant_recreate,
        )
        results = dense_retrieve(
            corpus,
            queries,
            encoder,
            backend,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
    elif args.backend == "elasticsearch":
        hosts = [h.strip() for h in args.es_hosts.split(",") if h.strip()]
        encoder = SentenceTransformerEncoder(args.model)
        backend = ElasticsearchVectorBackend(args.es_index, hosts=hosts, delete_if_exists=not args.no_es_delete)
        results = dense_retrieve(
            corpus,
            queries,
            encoder,
            backend,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
    else:
        raise SystemExit(f"Unknown backend {args.backend!r}")

    ndcg, map_, recall, precision = evaluate_results(qrels, results, k_values)
    logger.info("NDCG: %s", ndcg)
    logger.info("MAP: %s", map_)
    logger.info("Recall: %s", recall)
    logger.info("Precision: %s", precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
