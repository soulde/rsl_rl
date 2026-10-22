"""Resolve local AMP motion paths and supported dataset Hub URLs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def resolve_motion_dir(source: str | Path) -> str:
    """Return a local directory for a path or ModelScope/Hugging Face repo URL.

    Hub snapshots are stored in each client's normal cache. URLs may point to
    either a model or dataset repository; local filesystem paths pass through.
    """
    source = str(source)
    parsed = urlparse(source)
    if not parsed.scheme:
        return source
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported AMP motion source scheme {parsed.scheme!r}; use a local path, "
            "ModelScope URL, or Hugging Face URL."
        )

    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if host in {"modelscope.cn", "www.modelscope.cn"}:
        hub = "modelscope"
    elif host in {"huggingface.co", "www.huggingface.co", "hf.co"}:
        hub = "huggingface"
    else:
        raise ValueError(f"Unsupported AMP motion Hub URL host: {host or '<missing>'}")

    repo_type = "model"
    if parts and parts[0] in {"datasets", "models"}:
        repo_type = "dataset" if parts[0] == "datasets" else "model"
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"AMP motion Hub URL must include an owner and repository: {source}")
    repo_id = "/".join(parts[:2])
    revision = None
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        revision = parts[3]

    if hub == "modelscope":
        try:
            from modelscope_hub import HubApi
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope AMP motion URLs require the 'modelscope-hub' package."
            ) from exc
        snapshot = HubApi().download_repo(
            repo_id,
            repo_type,
            revision=revision,
            allow_patterns=["*.npz"],
        )
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face AMP motion URLs require the 'huggingface-hub' package."
            ) from exc
        snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=["*.npz"],
        )

    return str(Path(snapshot).expanduser().resolve())
