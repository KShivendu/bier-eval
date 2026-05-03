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
