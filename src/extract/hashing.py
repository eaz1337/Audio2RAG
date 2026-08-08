"""File hashing — the basis of `doc_id`, used across every output artifact (see CLAUDE.md
"Idempotency": `doc_id = sha256(audio bytes)[:16]`)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB, so large recordings don't need to fit in memory at once


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_doc_id(path: Path) -> str:
    return compute_sha256(path)[:16]
