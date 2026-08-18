"""SMOKE-5 one-off runner (TASKS.md) — exploration only, not part of `src/`.

Reads eval/smoke_questions.yaml, runs each question through the `ask` CLI command, and
prints the result next to the expected citation window so you can eyeball whether
retrieval is "close enough to harden" or off.

The shipped smoke_questions.yaml points at tests/fixtures/sample_a.wav / sample_b.wav
with synthetic segments seeded below (SEED_DOCS) rather than real recordings, so this
run only proves the harness mechanics work end to end. Replace SEED_DOCS/skip the
seeding step and point smoke_questions.yaml at your own doc_id once you've run
`audio2rag ingest` on recordings you care about — that's the real SMOKE-5 pass.

Usage: python eval/run_smoke_questions.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import cli  # noqa: E402
from extract.hashing import compute_sha256  # noqa: E402
from fakes import FakeASRBackend, FakeEmbedder, FakeLLM  # noqa: E402
from load.vector_store import write_chunks  # noqa: E402
from models.schemas import Chunk, Segment, TranscriptType  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "smoke_questions.yaml"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

FAKE_CONFIG = {
    "retrieval": {"dense_top_k": 30},
    "answer": {"refusal_threshold": 0.35, "llm": "fake"},
}

SEED_DOCS = [
    {
        "path": FIXTURES / "sample_a.wav",
        "title": "OS Lecture",
        "course": "Operating Systems",
        "texts": [
            (12.0, 16.2, "deadlock needs all four Coffman conditions"),
            (40.0, 45.0, "a deadlock occurs when processes wait on each other forever"),
            (60.0, 66.0, "a race condition is a timing bug, not a permanent stall"),
        ],
    },
    {
        "path": FIXTURES / "sample_b.wav",
        "title": "DB Lecture",
        "course": "Databases",
        "texts": [
            (10.0, 15.0, "ACID stands for atomicity consistency isolation durability"),
            (50.0, 56.0, "durability is guaranteed by writing to the transaction log before commit"),
        ],
    },
]


def _seed_fixture_store(output_dir: Path, store_dir: Path, embedder: FakeEmbedder) -> None:
    """Ingests SEED_DOCS and writes one chunk per question directly, following the
    pattern in tests/test_cli.py::TestAskCommand (embed_text == query, so the fake
    hash-derived embedder always ranks the right chunk first)."""
    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    chunks: list[Chunk] = []

    for doc in SEED_DOCS:
        doc_id = compute_sha256(doc["path"])[:16]
        segments = [
            Segment(doc_id=doc_id, seg=i, start=start, end=end, text=text, conf=0.9)
            for i, (start, end, text) in enumerate(doc["texts"])
        ]
        cli.ingest_file(
            doc["path"],
            backend=FakeASRBackend(segments),
            language="en",
            output_dir=output_dir,
            doc_type=TranscriptType.LECTURE,
            title=doc["title"],
            course=doc["course"],
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )
        doc_questions = [q for q in questions if q["doc_id"] == doc_id]
        for i, q in enumerate(doc_questions):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=i,
                    start=q["expected_start_s"],
                    end=q["expected_end_s"],
                    segment_ids=[0],
                    display_text=f"(fixture excerpt for: {q['question']})",
                    embed_text=q["question"],
                )
            )

    write_chunks(chunks, embedder, store_dir)


def main() -> None:
    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    embedder = FakeEmbedder(dim=8)
    llm = FakeLLM(response="[SMOKE-5 fixture answer — see cited excerpt]")
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp:
        output_dir, store_dir = Path(tmp) / "output", Path(tmp) / "store"
        cli.OUTPUT_DIR, cli.STORE_DIR = output_dir, store_dir
        _seed_fixture_store(output_dir, store_dir, embedder)
        cli.load_config = lambda: FAKE_CONFIG
        cli.build_embedder = lambda config: embedder
        cli.build_llm = lambda config: llm

        passed = 0
        for q in questions:
            result = runner.invoke(cli.app, ["ask", q["question"], "--json"])
            payload = json.loads(result.output) if result.exit_code == 0 else None
            refused = payload is not None and "citations" not in payload

            if q["expect_refusal"]:
                ok = refused
                actual = "refused" if refused else f"answered ({payload['citations'][0]})"
            elif refused:
                ok = False
                actual = f"refused ({payload['reason']})"
            else:
                citation = payload["citations"][0]
                ok = (
                    citation["doc_id"] == q["doc_id"]
                    and q["expected_start_s"] <= citation["start"] <= q["expected_end_s"]
                )
                actual = f"[{citation['title']}, {citation['start']:.1f}s-{citation['end']:.1f}s]"

            passed += ok
            status = "PASS" if ok else "FAIL"
            print(f"{status}  {q['question']!r}")
            print(f"      expected: doc={q['doc_id']} window={q['expected_start_s']}-{q['expected_end_s']}s refusal={q['expect_refusal']}")
            print(f"      actual:   {actual}")

    print(f"\n{passed}/{len(questions)} matched expectations (fixture run — not a real SMOKE-5 opinion, see module docstring)")


if __name__ == "__main__":
    main()
