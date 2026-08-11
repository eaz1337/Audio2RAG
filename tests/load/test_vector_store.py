from fakes import FakeEmbedder

from load.vector_store import collection_metadata, lance_dir, read_chunks, write_chunks
from models.schemas import Chunk

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


def test_multiple_writes_accumulate_rows(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk(chunk_id=0)], embedder, tmp_path)
    write_chunks([make_chunk(chunk_id=1)], embedder, tmp_path)

    rows = read_chunks(tmp_path)

    assert len(rows) == 2


def test_write_with_no_chunks_is_a_no_op(tmp_path):
    write_chunks([], FakeEmbedder(dim=8), tmp_path)

    assert collection_metadata(tmp_path) is None


def test_collection_metadata_records_model_name_and_dim(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks([make_chunk()], embedder, tmp_path)

    assert collection_metadata(tmp_path) == (embedder.name, 8)


def test_collection_metadata_is_none_before_any_write(tmp_path):
    assert collection_metadata(tmp_path) is None
