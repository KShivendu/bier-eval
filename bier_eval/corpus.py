from __future__ import annotations


def passage_text(doc: dict) -> str:
    """BEIR corpus entries use title + text."""
    title = (doc.get("title") or "").strip()
    body = (doc.get("text") or "").strip()
    if title and body:
        return f"{title} {body}"
    return title or body
