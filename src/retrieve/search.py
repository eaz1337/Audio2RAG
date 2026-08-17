"""Dense and BM25 sparse search over the local stores (spec.md §6, TASKS.md SEARCH-1,
SEARCH-2). `dense_search` alone is the Standard-tier deliverable: smart search, no LLM
call. `sparse_search` is the Advanced-tier lexical counterpart — exact-term matches (e.g.
rare technical terms) that a dense embedder can miss — fused with it in a later task
(CLAUDE.md: `retrieve/` is a read path and must never write to the store).

`filter_doc_ids` (SEARCH-5) restricts both to a set of doc_ids matching metadata filters
(`--course --type --after --before --speaker --tag`), applied *before* ranking so
filtering behaves identically whether or not fusion/reranking runs downstream."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as date_
from pathlib import Path

import bm25s

from load import bm25_index
from load.vector_store import assert_embedder_matches, open_table
from models.schemas import RetrievalHit, TranscriptMeta, TranscriptType
from transform.embed import Embedder


def filter_doc_ids(
    metas: Iterable[TranscriptMeta],
    *,
    course: str | None = None,
    type: TranscriptType | None = None,
    after: date_ | None = None,
    before: date_ | None = None,
    speaker: str | None = None,
    tag: str | None = None,
) -> set[str] | None:
    """Doc_ids whose `TranscriptMeta` satisfies every given filter, or `None` if no filter
    is set — the caller's cue to not restrict the candidate set at all. `after`/`before`
    are inclusive bounds on `meta.date`; a doc without a date never matches either bound.
    `speaker` matches a diarized label or a relabeled name (`meta.speakers`)."""
    if course is None and type is None and after is None and before is None and speaker is None and tag is None:
        return None

    matched: set[str] = set()
    for meta in metas:
        if course is not None and meta.course != course:
            continue
        if type is not None and meta.type != type:
            continue
        if after is not None and (meta.date is None or meta.date < after):
            continue
        if before is not None and (meta.date is None or meta.date > before):
            continue
        if tag is not None and tag not in meta.tags:
            continue
        if speaker is not None and speaker not in meta.speakers and speaker not in meta.speakers.values():
            continue
        matched.add(meta.doc_id)
    return matched


def _doc_ids_where(doc_ids: set[str]) -> str:
    quoted = ", ".join(f"'{doc_id}'" for doc_id in sorted(doc_ids))
    return f"doc_id IN ({quoted})"


def dense_search(
    query: str,
    k: int,
    embedder: Embedder,
    store_dir: Path,
    doc_ids: set[str] | None = None,
) -> list[RetrievalHit]:
    """Embeds `query` and returns the `k` nearest chunks by cosine similarity, best match
    first. Raises `EmbeddingModelMismatch` if the store was built with a different embedder
    (CLAUDE.md "Model pinning"). Empty list if the store has no chunks yet, or if `doc_ids`
    is given and empty (every candidate filtered out)."""
    assert_embedder_matches(store_dir, embedder)
    table = open_table(store_dir)
    if table is None:
        return []
    if doc_ids is not None and not doc_ids:
        return []

    [vector] = embedder.embed([query])
    search = table.search(vector.tolist()).metric("cosine")
    if doc_ids is not None:
        search = search.where(_doc_ids_where(doc_ids))
    rows = search.limit(k).to_list()

    return [
        RetrievalHit(
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            score=1.0 - row["_distance"],
            start=row["start"],
            end=row["end"],
            segment_ids=row["segment_ids"],
            display_text=row["display_text"],
        )
        for row in rows
    ]


def sparse_search(
    query: str, k: int, store_dir: Path, doc_ids: set[str] | None = None
) -> list[RetrievalHit]:
    """BM25 lexical search over the same chunks the dense index holds. Empty list if the
    BM25 index has no chunks yet (SEARCH-2), or if `doc_ids` is given and empty. No
    embedder needed — no network call, no model download, purely local."""
    rows = bm25_index.read_rows(store_dir)
    if not rows:
        return []
    if doc_ids is not None and not doc_ids:
        return []

    directory = bm25_index.bm25_dir(store_dir)
    retriever = bm25s.BM25.load(directory, load_corpus=False, show_progress=False)
    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    # When filtering, rank the whole corpus first so the top-k slice below is taken after
    # filtering, not before — otherwise a legitimate hit could be pushed out by rows that
    # will be dropped anyway.
    query_k = len(rows) if doc_ids is not None else min(k, len(rows))
    [hit_rows], [scores] = retriever.retrieve(
        query_tokens, corpus=rows, k=query_k, show_progress=False
    )

    hits = [
        RetrievalHit(
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            score=float(score),
            start=row["start"],
            end=row["end"],
            segment_ids=row["segment_ids"],
            display_text=row["display_text"],
        )
        for row, score in zip(hit_rows, scores, strict=True)
        if doc_ids is None or row["doc_id"] in doc_ids
    ]
    return hits[:k]
