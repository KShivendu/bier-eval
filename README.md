# bier-eval

Small toolkit for [BEIR](https://github.com/beir-cellar/beir)-style retrieval: dense encoders (SentenceTransformers) plus FAISS, Qdrant, Elasticsearch, or Tantivy (lexical BM25).

## Install

```bash
pip install -e .
# optional backends / encoder stack
pip install -e ".[faiss]"          # default dense path
pip install -e ".[qdrant]"
pip install -e ".[elasticsearch]"
pip install -e ".[tantivy]"
pip install -e ".[encoders]"       # sentence-transformers + torch
pip install -e ".[all]"
```

## Run

```bash
bier-eval --dataset scifact --backend faiss
```

`--dataset` takes a [BEIR dataset name](https://github.com/beir-cellar/beir#beers-available-datasets) (see the upstream table for all options). Downloads go under `datasets/` (override with `--data-dir`). For Qdrant/Elasticsearch, point `--qdrant-url` / `--es-hosts` at a running instance.

### Reference result (SciFact, Tantivy BM25)

```bash
bier-eval --dataset scifact --backend tantivy
```

SciFact **test** (300 queries, 5,183 documents), default `--top-k` 100 and `--k-values` 1,3,5,10,100:

| Metric | @1 | @3 | @5 | @10 | @100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NDCG | 0.527 | 0.611 | 0.630 | 0.656 | 0.678 |
| MAP | 0.513 | 0.586 | 0.597 | 0.610 | 0.615 |
| Recall | 0.513 | 0.667 | 0.711 | 0.786 | 0.881 |

### Reference result (SciFact, FAISS + MiniLM)

```bash
pip install -e ".[faiss,encoders]"
bier-eval --dataset scifact --backend faiss
```

Same split and defaults as above; dense model `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP`, cosine via inner product on normalized vectors:

| Metric | @1 | @3 | @5 | @10 | @100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NDCG | 0.503 | 0.597 | 0.629 | 0.645 | 0.677 |
| MAP | 0.482 | 0.566 | 0.588 | 0.596 | 0.603 |
| Recall | 0.482 | 0.660 | 0.738 | 0.783 | 0.925 |

Figures are from single representative runs; minor drift is possible across `tantivy`, `faiss-cpu`, `sentence-transformers`, torch, and BEIR versions.
