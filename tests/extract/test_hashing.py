import hashlib
import shutil
from pathlib import Path

from extract.hashing import compute_doc_id

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_A = FIXTURES / "sample_a.wav"
SAMPLE_B = FIXTURES / "sample_b.wav"


def test_matches_sha256_prefix_of_bytes():
    expected = hashlib.sha256(SAMPLE_A.read_bytes()).hexdigest()[:16]

    assert compute_doc_id(SAMPLE_A) == expected


def test_stable_across_runs():
    assert compute_doc_id(SAMPLE_A) == compute_doc_id(SAMPLE_A)


def test_differs_for_different_content():
    assert compute_doc_id(SAMPLE_A) != compute_doc_id(SAMPLE_B)


def test_same_content_different_filename_yields_same_id(tmp_path):
    renamed = tmp_path / "a_different_name.wav"
    shutil.copyfile(SAMPLE_A, renamed)

    assert compute_doc_id(renamed) == compute_doc_id(SAMPLE_A)
