"""The directory of models a worker serves from, named by their SHA-256."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

GGUF_MAGIC = b"GGUF"
CHUNK_BYTES = 8 << 20
DIGEST = re.compile(r"[0-9a-f]{64}")


def default_model_dir() -> Path:
    configured = os.environ.get("SHARDLINE_MODEL_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "shardline" / "models"


def model_path(model_dir: Path, digest: str) -> Path:
    if not DIGEST.fullmatch(digest):
        raise ValueError("a model is named by its SHA-256: 64 lowercase hex digits")
    return model_dir / f"{digest}.gguf"


def add_model(source: Path, model_dir: Path) -> str:
    """Copies a GGUF file into model_dir as <sha256>.gguf and returns the digest.

    The file is hashed while it is copied and only renamed into place when the copy
    is complete, so a worker never opens a half-written model.
    """
    digest = hashlib.sha256()
    model_dir.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader:
        if reader.read(len(GGUF_MAGIC)) != GGUF_MAGIC:
            raise ValueError(f"{source} is not a GGUF file")
        reader.seek(0)
        handle, temporary = tempfile.mkstemp(dir=model_dir, suffix=".partial")
        try:
            with os.fdopen(handle, "wb") as writer:
                while chunk := reader.read(CHUNK_BYTES):
                    digest.update(chunk)
                    writer.write(chunk)
            destination = model_path(model_dir, digest.hexdigest())
            os.replace(temporary, destination)
            destination.chmod(0o644)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
