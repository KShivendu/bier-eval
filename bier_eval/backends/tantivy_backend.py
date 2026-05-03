from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Characters reserved by Tantivy's query parser (claims often contain '(+)', quotes, etc.).
_TANTIVY_RESERVED = frozenset("+-&|!(){}[]^\"~*?:/'")


def _escape_tantivy_query(q: str) -> str:
    out: list[str] = []
    for ch in q:
        if ch == "\\":
            out.append("\\\\")
        elif ch in _TANTIVY_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


class TantivyLexicalBackend:
    """In-process Tantivy full-text index (BM25-style ranking). No embeddings."""

    def __init__(self, *, index_path: str | None = None) -> None:
        import tantivy

        self._tantivy = tantivy
        self.index_path = index_path
        self._index = None

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        tantivy = self._tantivy
        schema = (
            tantivy.SchemaBuilder()
            .add_text_field("doc_id", stored=True, index_option="basic")
            .add_text_field("text", stored=False)
            .build()
        )
        self._index = tantivy.Index(schema, path=self.index_path)
        writer = self._index.writer()
        for doc_id, text in zip(doc_ids, texts):
            d = tantivy.Document()
            d.add_text("doc_id", doc_id)
            d.add_text("text", text)
            writer.add_document(d)
        writer.commit()
        if self.index_path is None:
            writer.wait_merging_threads()
        self._index.reload()
        logger.info("Tantivy indexed %d documents", len(doc_ids))

    def search(self, queries: list[str], *, top_k: int) -> list[list[tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("build() first")
        searcher = self._index.searcher()
        out: list[list[tuple[str, float]]] = []
        for q in queries:
            q = (q or "").strip()
            if not q:
                out.append([])
                continue
            safe = _escape_tantivy_query(q)
            try:
                parsed = self._index.parse_query(safe, ["text"])
            except ValueError as e:
                logger.warning("Tantivy parse failed for query; returning no hits: %s (%s)", q[:120], e)
                out.append([])
                continue
            sr = searcher.search(parsed, top_k)
            hits: list[tuple[str, float]] = []
            for score, addr in sr.hits:
                doc = searcher.doc(addr)
                raw = doc["doc_id"]
                did = raw[0] if isinstance(raw, list) and raw else raw
                hits.append((str(did), float(score)))
            out.append(hits)
        return out
