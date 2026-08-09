"""Directory discovery for `ingest <dir> --recursive` (see CLAUDE.md pipeline layout:
`extract/` owns file discovery, format + path validation, hashing)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_audio_files(
    directory: Path, supported_extensions: list[str], *, recursive: bool = False
) -> list[Path]:
    """Finds files under `directory` whose extension is in `supported_extensions`
    (case-insensitive, dot optional). Files with an unsupported extension are skipped with
    a warning rather than raised as an error."""
    extensions = {ext.lower().lstrip(".") for ext in supported_extensions}
    pattern = "**/*" if recursive else "*"

    found: list[Path] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") in extensions:
            found.append(path)
        else:
            logger.warning("skipping unsupported file: %s", path)
    return found
