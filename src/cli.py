"""CLI entry point (see CLAUDE.md pipeline layout). Reads `config.yaml` to select the
configured `ASRBackend` and wires it through `Transcriber` onto the canonical write path —
nothing here bypasses `ASRBackend` or writes artifacts outside `load/write_canonical`."""

from __future__ import annotations

from datetime import UTC, date as date_, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from extract.discover import discover_audio_files
from extract.hashing import compute_sha256
from load.render_md import render_md_from_jsonl
from load.render_pdf import render_pdf_from_jsonl
from load.render_srt import render_srt_from_jsonl
from load.write_canonical import segments_path, write_canonical
from models.schemas import Segment, TranscriptMeta, TranscriptType
from transform.asr.base import ASRBackend
from transform.transcribe import Transcriber

app = typer.Typer()

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("output")

_RENDERERS = {
    "pdf": render_pdf_from_jsonl,
    "md": render_md_from_jsonl,
    "srt": render_srt_from_jsonl,
}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def build_asr_backend(config: dict[str, Any]) -> ASRBackend:
    backend_name = config["asr"]["backend"]
    if backend_name == "assemblyai":
        from transform.asr.hosted import AssemblyAIBackend

        return AssemblyAIBackend()
    raise ValueError(f"ASR backend {backend_name!r} is not implemented yet")


def _parse_render_targets(render: str, default: list[str]) -> list[str]:
    if not render:
        return list(default)
    targets = [token.strip() for token in render.split(",") if token.strip()]
    unknown = [target for target in targets if target not in _RENDERERS]
    if unknown:
        raise typer.BadParameter(
            f"unknown renderer(s) {unknown}, choose from {sorted(_RENDERERS)}"
        )
    return targets


def _map_speakers(segments: list[Segment], names: list[str]) -> dict[str, str]:
    """Maps `--speaker` names, in the order given, onto diarized labels in the order they
    first appear — real name mapping by label is `relabel`'s job (INGEST-4)."""
    labels: list[str] = []
    for segment in segments:
        if segment.speaker and segment.speaker not in labels:
            labels.append(segment.speaker)
    return dict(zip(labels, names, strict=False))


def ingest_file(
    path: Path,
    *,
    backend: ASRBackend,
    language: str,
    output_dir: Path,
    doc_type: TranscriptType,
    title: str | None,
    course: str | None,
    date: date_ | None,
    speakers: list[str],
    tags: list[str],
    render_targets: list[str],
    asr_model_name: str,
    force: bool = False,
) -> Path:
    transcriber = Transcriber(backend)
    segments = transcriber.transcribe_cached(path, language, output_dir, force=force)

    sha256 = compute_sha256(path)
    doc_id = sha256[:16]

    meta = TranscriptMeta(
        doc_id=doc_id,
        source_path=str(path),
        sha256=sha256,
        type=doc_type,
        title=title or path.stem,
        course=course,
        speakers=_map_speakers(segments, speakers),
        date=date,
        duration_s=max((segment.end for segment in segments), default=0.0),
        language=language,
        model=asr_model_name,
        tags=tags,
        ingested_at=datetime.now(UTC),
    )
    write_canonical(segments, meta, output_dir)

    for target in render_targets:
        _RENDERERS[target](doc_id, output_dir)

    return segments_path(doc_id, output_dir)


def ingest_directory(
    directory: Path,
    *,
    recursive: bool,
    supported_extensions: list[str],
    **ingest_file_kwargs: Any,
) -> list[Path]:
    """Ingests every supported file under `directory`. A single file's failure is reported
    and skipped rather than aborting the batch (TASKS.md INGEST-3)."""
    files = discover_audio_files(directory, supported_extensions, recursive=recursive)

    results: list[Path] = []
    for file_path in files:
        try:
            results.append(ingest_file(file_path, **ingest_file_kwargs))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            typer.echo(f"skipping {file_path}: {exc}", err=True)
    return results


@app.callback()
def main() -> None:
    """audio2rag — turn recordings into a queryable knowledge base."""


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True),
    recursive: bool = typer.Option(
        False, "--recursive", help="Recurse into subdirectories when PATH is a directory."
    ),
    doc_type: TranscriptType = typer.Option(TranscriptType.OTHER, "--type"),
    title: str | None = typer.Option(None, "--title"),
    course: str | None = typer.Option(None, "--course"),
    date: str | None = typer.Option(None, "--date"),
    speaker: list[str] = typer.Option([], "--speaker"),
    tag: list[str] = typer.Option([], "--tag"),
    render: str = typer.Option("", "--render"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Transcribes `path` (or reuses the cached transcript) and writes the canonical
    artifacts plus any `--render` targets. Values from the metadata flags land in
    `<doc_id>.meta.json`; renderers stay opt-in. If `path` is a directory, every supported
    file in it is ingested (`--recursive` to descend into subdirectories); an unsupported
    file is skipped with a warning and a single file's failure does not abort the batch."""
    config = load_config()
    backend = build_asr_backend(config)
    render_targets = _parse_render_targets(render, config["ingest"]["render_default"])

    common_kwargs: dict[str, Any] = dict(
        backend=backend,
        language=config["asr"]["language"],
        output_dir=OUTPUT_DIR,
        doc_type=doc_type,
        title=title,
        course=course,
        date=date_.fromisoformat(date) if date else None,
        speakers=speaker,
        tags=tag,
        render_targets=render_targets,
        asr_model_name=config["asr"]["backend"],
        force=force,
    )

    if path.is_dir():
        ingest_directory(
            path,
            recursive=recursive,
            supported_extensions=config["ingest"]["supported_extensions"],
            **common_kwargs,
        )
        return

    ingest_file(path, **common_kwargs)


if __name__ == "__main__":
    app()
