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
from load.manage import list_transcripts, relabel_speakers, remove_transcript
from load.render_md import render_md_from_jsonl
from load.render_pdf import render_pdf_from_jsonl
from load.render_srt import render_srt_from_jsonl
from load.vector_store import delete_doc, reindex_all
from load.write_canonical import meta_path, segments_path, write_canonical
from models.schemas import Segment, TranscriptMeta, TranscriptType
from retrieve.search import dense_search, filter_doc_ids
from transform.asr.base import ASRBackend
from transform.embed import Embedder
from transform.transcribe import Transcriber

app = typer.Typer()

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("output")
STORE_DIR = Path("store")

_RENDERERS = {
    "pdf": render_pdf_from_jsonl,
    "md": render_md_from_jsonl,
    "srt": render_srt_from_jsonl,
}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def build_asr_backend(config: dict[str, Any], *, diarize: bool = False) -> ASRBackend:
    backend_name = config["asr"]["backend"]
    if backend_name == "assemblyai":
        from transform.asr.hosted import AssemblyAIBackend

        return AssemblyAIBackend()
    if backend_name == "whisper-local":
        from transform.asr.whisper_local import WhisperLocalBackend

        whisper_config = config["asr"]["whisper_local"]
        return WhisperLocalBackend(
            model=whisper_config["model"],
            vad_filter=whisper_config["vad_filter"],
            word_timestamps=whisper_config["word_timestamps"],
            condition_on_previous_text_max_minutes=whisper_config[
                "condition_on_previous_text_max_minutes"
            ],
            diarize=diarize or whisper_config.get("diarize", False),
        )
    raise ValueError(f"ASR backend {backend_name!r} is not implemented yet")


def build_embedder(config: dict[str, Any]) -> Embedder:
    from transform.embed import BgeM3Embedder

    return BgeM3Embedder(model_name=config["embedding"]["model"])


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
        diarized=any(segment.speaker for segment in segments),
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
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Diarize with pyannote (whisper-local only; hosted backends diarize inline).",
    ),
) -> None:
    """Transcribes `path` (or reuses the cached transcript) and writes the canonical
    artifacts plus any `--render` targets. Values from the metadata flags land in
    `<doc_id>.meta.json`; renderers stay opt-in. If `path` is a directory, every supported
    file in it is ingested (`--recursive` to descend into subdirectories); an unsupported
    file is skipped with a warning and a single file's failure does not abort the batch."""
    config = load_config()
    backend = build_asr_backend(config, diarize=diarize)
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


def _format_duration(duration_s: float) -> str:
    minutes, seconds = divmod(int(duration_s), 60)
    return f"{minutes}m{seconds:02d}s"


@app.command(name="list")
def list_command() -> None:
    """Prints doc_id, title, type, and duration for every ingested transcript."""
    metas = list_transcripts(OUTPUT_DIR)
    for meta in metas:
        typer.echo(
            f"{meta.doc_id}\t{meta.title}\t{meta.type.value}\t{_format_duration(meta.duration_s)}"
        )


@app.command()
def rm(doc_id: str = typer.Argument(...)) -> None:
    """Removes every canonical and rendered artifact for `doc_id`, plus its chunks from
    the vector store."""
    if not meta_path(doc_id, OUTPUT_DIR).exists():
        typer.echo(f"no transcript with doc_id {doc_id!r}", err=True)
        raise typer.Exit(code=1)
    remove_transcript(doc_id, OUTPUT_DIR)
    delete_doc(STORE_DIR, doc_id)


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@app.command()
def search(
    query: str = typer.Argument(...),
    k: int = typer.Option(5, "--k"),
    course: str | None = typer.Option(None, "--course"),
    doc_type: TranscriptType | None = typer.Option(None, "--type"),
    after: str | None = typer.Option(None, "--after"),
    before: str | None = typer.Option(None, "--before"),
    speaker: str | None = typer.Option(None, "--speaker"),
    tag: str | None = typer.Option(None, "--tag"),
) -> None:
    """[STANDARD] Dense search over the local vector store: prints the `k` best-matching
    chunks, best first, as `[title, HH:MM:SS-HH:MM:SS]  score  text`. No LLM call — this is
    smart search, not answering. `--course/--type/--after/--before/--speaker/--tag` restrict
    the candidate set before ranking (TASKS.md SEARCH-5)."""
    config = load_config()
    embedder = build_embedder(config)
    metas = list_transcripts(OUTPUT_DIR)
    titles = {meta.doc_id: meta.title for meta in metas}
    doc_ids = filter_doc_ids(
        metas,
        course=course,
        type=doc_type,
        after=date_.fromisoformat(after) if after else None,
        before=date_.fromisoformat(before) if before else None,
        speaker=speaker,
        tag=tag,
    )

    hits = dense_search(query, k, embedder, STORE_DIR, doc_ids=doc_ids)
    for hit in hits:
        title = titles.get(hit.doc_id, hit.doc_id)
        timestamp = f"{_format_timestamp(hit.start)}-{_format_timestamp(hit.end)}"
        typer.echo(f"[{title}, {timestamp}]\t{hit.score:.3f}\t{hit.display_text}")


@app.command()
def reindex() -> None:
    """Rebuilds the vector store from `output/*.segments.jsonl` for every ingested
    document, without re-transcribing. Run after an embedding-model or chunking-config
    change — `EmbeddingModelMismatch` at query time points here."""
    config = load_config()
    embedder = build_embedder(config)
    chunking = config["chunking"]
    doc_ids = reindex_all(
        OUTPUT_DIR,
        STORE_DIR,
        embedder,
        target_tokens=chunking["target_tokens"],
        overlap_ratio=chunking["overlap_ratio"],
        pause_threshold_s=chunking["pause_threshold_s"],
    )
    typer.echo(f"reindexed {len(doc_ids)} document(s)")


def _parse_speaker_labels(pairs: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"expected LABEL=NAME, got {pair!r}")
        label, name = pair.split("=", 1)
        labels[label.strip()] = name.strip()
    return labels


@app.command()
def relabel(
    doc_id: str = typer.Argument(...),
    speaker: list[str] = typer.Option(
        [], "--speaker", help="LABEL=NAME, e.g. --speaker SPEAKER_00='Dr. Kowalski'"
    ),
) -> None:
    """Edits `<doc_id>.meta.json`'s speaker labels only; never re-runs transcription."""
    if not meta_path(doc_id, OUTPUT_DIR).exists():
        typer.echo(f"no transcript with doc_id {doc_id!r}", err=True)
        raise typer.Exit(code=1)
    labels = _parse_speaker_labels(speaker)
    relabel_speakers(doc_id, OUTPUT_DIR, labels)


if __name__ == "__main__":
    app()
