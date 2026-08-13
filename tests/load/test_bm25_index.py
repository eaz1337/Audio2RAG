from load.bm25_index import bm25_dir, delete_doc, drop_index, read_rows, write_chunks
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
        "embed_text": "[Lecture 3, 00:00:00-00:00:04]\nDeadlock occurs when all four Coffman conditions hold.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_write_creates_bm25_dir(tmp_path):
    write_chunks([make_chunk()], tmp_path)

    assert bm25_dir(tmp_path).exists()


def test_read_rows_returns_empty_list_before_any_write(tmp_path):
    assert read_rows(tmp_path) == []


def test_written_rows_carry_every_chunk_field(tmp_path):
    chunk = make_chunk(chunk_id=1, segment_ids=[2, 3, 4])
    write_chunks([chunk], tmp_path)

    [row] = read_rows(tmp_path)

    assert row["doc_id"] == chunk.doc_id
    assert row["chunk_id"] == chunk.chunk_id
    assert row["start"] == chunk.start
    assert row["end"] == chunk.end
    assert row["segment_ids"] == chunk.segment_ids
    assert row["display_text"] == chunk.display_text


def test_read_rows_filters_by_doc_id(tmp_path):
    write_chunks([make_chunk(chunk_id=0)], tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], tmp_path)

    rows = read_rows(tmp_path, doc_id=OTHER_DOC_ID)

    assert len(rows) == 1
    assert rows[0]["doc_id"] == OTHER_DOC_ID


def test_writes_for_different_doc_ids_accumulate_rows(tmp_path):
    write_chunks([make_chunk(chunk_id=0)], tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], tmp_path)

    assert len(read_rows(tmp_path)) == 2


def test_reingesting_same_doc_id_leaves_row_count_unchanged(tmp_path):
    chunks = [make_chunk(chunk_id=0), make_chunk(chunk_id=1, segment_ids=[2, 3])]
    write_chunks(chunks, tmp_path)
    write_chunks(chunks, tmp_path)

    assert len(read_rows(tmp_path)) == 2


def test_reingesting_with_fewer_chunks_leaves_no_orphans(tmp_path):
    write_chunks(
        [make_chunk(chunk_id=0), make_chunk(chunk_id=1), make_chunk(chunk_id=2)], tmp_path
    )
    write_chunks([make_chunk(chunk_id=0)], tmp_path)

    assert len(read_rows(tmp_path)) == 1


def test_reingesting_one_doc_id_does_not_touch_another(tmp_path):
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], tmp_path)
    write_chunks([make_chunk(chunk_id=0), make_chunk(chunk_id=1)], tmp_path)
    write_chunks([make_chunk(chunk_id=0)], tmp_path)

    rows = read_rows(tmp_path, doc_id=OTHER_DOC_ID)
    assert len(rows) == 1


def test_delete_doc_removes_only_that_doc_id(tmp_path):
    write_chunks([make_chunk(chunk_id=0)], tmp_path)
    write_chunks([make_chunk(doc_id=OTHER_DOC_ID, chunk_id=0)], tmp_path)

    delete_doc(tmp_path, DOC_ID)

    rows = read_rows(tmp_path)
    assert [row["doc_id"] for row in rows] == [OTHER_DOC_ID]


def test_delete_doc_is_a_no_op_before_any_write(tmp_path):
    delete_doc(tmp_path, DOC_ID)  # must not raise

    assert read_rows(tmp_path) == []


def test_delete_last_doc_removes_the_index_directory(tmp_path):
    write_chunks([make_chunk()], tmp_path)

    delete_doc(tmp_path, DOC_ID)

    assert not bm25_dir(tmp_path).exists()


def test_write_with_no_chunks_is_a_no_op(tmp_path):
    write_chunks([], tmp_path)

    assert read_rows(tmp_path) == []
    assert not bm25_dir(tmp_path).exists()


def test_drop_index_removes_the_directory(tmp_path):
    write_chunks([make_chunk()], tmp_path)

    drop_index(tmp_path)

    assert not bm25_dir(tmp_path).exists()
    assert read_rows(tmp_path) == []


def test_drop_index_is_a_no_op_before_any_write(tmp_path):
    drop_index(tmp_path)  # must not raise
