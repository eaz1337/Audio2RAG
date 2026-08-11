from datetime import UTC, datetime

import pytest
from fakes import FakeEmbedder

from load.vector_store import (
    EmbeddingModelMismatch,
    assert_embedder_matches,
    collection_metadata,
    delete_doc,
    lance_dir,
    read_chunks,
    reindex_all,
    write_chunks,
)
from load.write_canonical import write_canonical
from models.schemas import Chunk, Segment, TranscriptMeta, TranscriptType

DOC_ID = "a3f9c1b2d4e6f801"
OTHER_DOC_ID = "b1c2d3e4f5061708"


def make_chunk(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "chunk_id": 0,
        "start": 0.0,
        "end": 4.2,
        "segment_ids": [0, 1],
        "display_text": "Deadlock occurs when all four Coffman conditions hold.",
        "embed_text": "[Lecture 3, 00:00:00–00:00:04]\nDeadlock occurs when all four Coffman conditions hold.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_write_creates_lance_dir(tmp_path):
    write_chunks([make_chunk()], FakeEmbedder(dim=8), tmp_path)

    assert lance_dir(tmp_path).exists()


def test_written_rows_carry_every_chunk_field(tmp_path):
    chunk = make_chunk(chunk_id=1, segment_ids=[2, 3, 4])
    write_chunks([chunk], FakeEmbedder(dim=8), tmp_path)

    rows = read_chunks(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == chunk.doc_id
    assert row["chunk_id"] == chunk.chunk_id
    assert row["start"] == chunk.start
    assert row["end"] == chunk.end
    assert row["segment_ids"] == chunk.segment_ids
    assert row["display_text"] == chunk.display_text
    assert row["embed_text"] == chunk.embed_text
    assert len(row["vector"]) == 8


def test_vector_matches_embedder_output_for_embed_text(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)

    [row] = read_chunks(tmp_path)
    expected = embedder.embed([chunk.embed_text])[0]

    assert list(row["vector"]) == list(expected)


def test_read_chunks_filters_by_doc_id(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], embedder, tmp_path)

    rows = read_chunks(tmp_path, doc_id=OTHER_DOC_ID)

    assert len(rows) == 1
    assert rows[0]["doc_id"] == OTHER_DOC_ID


def test_read_chunks_returns_empty_list_before_any_write(tmp_path):
    assert read_chunks(tmp_path) == []


def test_writes_for_different_doc_ids_accumulate_rows(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], embedder, tmp_path)

    rows = read_chunks(tmp_path)

    assert len(rows) == 2


def test_reingesting_same_doc_id_leaves_chunk_count_unchanged(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunks = [make_chunk(chunk_id=0), make_chunk(chunk_id=1, segment_ids=[2, 3])]
    write_chunks(chunks, embedder, tmp_path)
    write_chunks(chunks, embedder, tmp_path)

    rows = read_chunks(tmp_path)

    assert len(rows) == 2


def test_reingesting_with_fewer_chunks_leaves_no_orphans(tmp_path):
    """Simulates a chunk-size config change producing a different chunk count for the
    same doc_id — the old rows must not survive alongside the new ones (TASKS.md INDEX-3)."""
    embedder = FakeEmbedder(dim=8)
    write_chunks(
        [make_chunk(chunk_id=0), make_chunk(chunk_id=1), make_chunk(chunk_id=2)],
        embedder,
        tmp_path,
    )
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)

    rows = read_chunks(tmp_path)

    assert len(rows) == 1


def test_reingesting_one_doc_id_does_not_touch_another(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], embedder, tmp_path)
    write_chunks([make_chunk(chunk_id=0), make_chunk(chunk_id=1)], embedder, tmp_path)
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)

    rows = read_chunks(tmp_path, doc_id=OTHER_DOC_ID)

    assert len(rows) == 1


def test_delete_doc_removes_only_that_doc_id(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], embedder, tmp_path)

    delete_doc(tmp_path, DOC_ID)

    rows = read_chunks(tmp_path)
    assert [row["doc_id"] for row in rows] == [OTHER_DOC_ID]


def test_delete_doc_is_a_no_op_before_any_write(tmp_path):
    delete_doc(tmp_path, DOC_ID)  # must not raise

    assert read_chunks(tmp_path) == []


def test_write_with_no_chunks_is_a_no_op(tmp_path):
    write_chunks([], FakeEmbedder(dim=8), tmp_path)

    assert collection_metadata(tmp_path) is None


def test_collection_metadata_records_model_name_and_dim(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk()], embedder, tmp_path)

    assert collection_metadata(tmp_path) == (embedder.name, 8)


def test_collection_metadata_is_none_before_any_write(tmp_path):
    assert collection_metadata(tmp_path) is None


def make_meta(doc_id: str, **overrides) -> TranscriptMeta:
    fields = {
        "doc_id": doc_id,
        "source_path": "/tmp/audio.wav",
        "sha256": "a" * 64,
        "type": TranscriptType.LECTURE,
        "title": "Lecture 1",
        "course": None,
        "speakers": {},
        "date": None,
        "duration_s": 12.0,
        "language": "pl",
        "model": "fake",
        "tags": [],
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    fields.update(overrides)
    return TranscriptMeta(**fields)


def make_transcript_segment(doc_id: str, **overrides) -> Segment:
    fields = {
        "doc_id": doc_id,
        "seg": 0,
        "start": 0.0,
        "end": 4.2,
        "text": "Deadlock occurs when all four Coffman conditions hold.",
        "speaker": None,
        "conf": 0.9,
    }
    fields.update(overrides)
    return Segment(**fields)


def write_transcript(output_dir, doc_id: str, **meta_overrides) -> None:
    write_canonical(
        [make_transcript_segment(doc_id)], make_meta(doc_id, **meta_overrides), output_dir
    )


class TestAssertEmbedderMatches:
    def test_is_a_noop_before_any_write(self, tmp_path):
        assert_embedder_matches(tmp_path, FakeEmbedder(dim=8))  # must not raise

    def test_passes_when_model_and_dim_match(self, tmp_path):
        write_chunks([make_chunk()], FakeEmbedder(dim=8), tmp_path)

        assert_embedder_matches(tmp_path, FakeEmbedder(dim=8))  # must not raise

    def test_raises_on_dim_mismatch(self, tmp_path):
        write_chunks([make_chunk()], FakeEmbedder(dim=8), tmp_path)

        with pytest.raises(EmbeddingModelMismatch) as excinfo:
            assert_embedder_matches(tmp_path, FakeEmbedder(dim=16))

        message = str(excinfo.value)
        assert "8" in message
        assert "16" in message
        assert "reindex" in message


class TestReindexAll:
    def test_rebuilds_from_jsonl_for_every_doc(self, tmp_path):
        output_dir = tmp_path / "output"
        store_dir = tmp_path / "store"
        write_transcript(output_dir, DOC_ID, title="Lecture 1")
        write_transcript(output_dir, OTHER_DOC_ID, title="Lecture 2")

        doc_ids = reindex_all(output_dir, store_dir, FakeEmbedder(dim=8))

        assert set(doc_ids) == {DOC_ID, OTHER_DOC_ID}
        rows = read_chunks(store_dir)
        assert {row["doc_id"] for row in rows} == {DOC_ID, OTHER_DOC_ID}

    def test_replaces_dimension_on_embedder_change(self, tmp_path):
        output_dir = tmp_path / "output"
        store_dir = tmp_path / "store"
        write_transcript(output_dir, DOC_ID)
        reindex_all(output_dir, store_dir, FakeEmbedder(dim=8))

        reindex_all(output_dir, store_dir, FakeEmbedder(dim=16))

        assert collection_metadata(store_dir) == (FakeEmbedder(dim=16).name, 16)
        rows = read_chunks(store_dir)
        assert len(rows[0]["vector"]) == 16

    def test_doc_removed_from_corpus_is_dropped_from_the_store(self, tmp_path):
        """The store is rebuilt from scratch (`drop_table`), not delete-then-insert per
        doc_id, so a doc_id removed from `output_dir` between reindexes leaves no orphan."""
        output_dir = tmp_path / "output"
        store_dir = tmp_path / "store"
        write_transcript(output_dir, DOC_ID)
        write_transcript(output_dir, OTHER_DOC_ID)
        reindex_all(output_dir, store_dir, FakeEmbedder(dim=8))

        (output_dir / f"{OTHER_DOC_ID}.segments.jsonl").unlink()
        (output_dir / f"{OTHER_DOC_ID}.meta.json").unlink()
        reindex_all(output_dir, store_dir, FakeEmbedder(dim=8))

        rows = read_chunks(store_dir)
        assert [row["doc_id"] for row in rows] == [DOC_ID]

    def test_empty_corpus_leaves_no_table(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        store_dir = tmp_path / "store"

        doc_ids = reindex_all(output_dir, store_dir, FakeEmbedder(dim=8))

        assert doc_ids == []
        assert collection_metadata(store_dir) is None
