"""Listing, removal, and speaker relabeling over the canonical artifacts (TASKS.md
INGEST-4). `relabel_speakers` only ever touches `meta.json` — it must never call an
`ASRBackend` or rewrite `segments.jsonl`."""

from __future__ import annotations

from pathlib import Path

from load.write_canonical import meta_path, read_meta, segments_path
from models.schemas import TranscriptMeta

_RENDER_EXTENSIONS = ("pdf", "md", "srt")


def list_transcripts(output_dir: Path) -> list[TranscriptMeta]:
    """Reads every `*.meta.json` under `output_dir`, sorted by doc_id."""
    return [
        read_meta(path.name.removesuffix(".meta.json"), output_dir)
        for path in sorted(output_dir.glob("*.meta.json"))
    ]


def remove_transcript(doc_id: str, output_dir: Path) -> None:
    """Removes the canonical artifacts and any rendered files for `doc_id`. Vector-store
    cleanup is the caller's job (see `cli.rm`, `load.vector_store.delete_doc`)."""
    segments_path(doc_id, output_dir).unlink(missing_ok=True)
    meta_path(doc_id, output_dir).unlink(missing_ok=True)
    for extension in _RENDER_EXTENSIONS:
        (output_dir / f"{doc_id}.{extension}").unlink(missing_ok=True)


def relabel_speakers(doc_id: str, output_dir: Path, labels: dict[str, str]) -> TranscriptMeta:
    """Merges `labels` (diarized label -> real name) into `meta.speakers` and rewrites only
    `meta.json`. Never re-transcribes and never touches `segments.jsonl`."""
    meta = read_meta(doc_id, output_dir)
    updated = meta.model_copy(update={"speakers": {**meta.speakers, **labels}})
    meta_path(doc_id, output_dir).write_text(updated.model_dump_json(indent=2))
    return updated
