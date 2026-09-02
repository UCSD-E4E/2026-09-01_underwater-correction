"""Resolving NAS-relative paths out of the manifest.

`Image.path` and `DiveSlate.path` are share-relative -- `2024.06.20.REEF/new
v-slate photos/SMILE vslate 1.pdf` -- rooted at the worker's
`e4e_nas.raw_root_path`, which on a dev box is the mounted share plus
`REEF/data`. Resolving them in place beats materializing a checksum-named
cache: the corpus is 38k frames and the raws are ~15 MB each.

The checksum cache is still supported and still takes precedence, because it
is the form that needs no mount at all.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_under_root"]


def resolve_under_root(root: Path | str | None, relative: str | None) -> Path | None:
    """Resolve a share-relative path under `root`, or None if absent.

    Returns None rather than raising for a missing file: on a partially-synced
    mount, a dive whose raws were never pulled is an ordinary outcome, and a
    sweep should record it and move on.

    Raises for a path that escapes the root. This is the one place a read-only
    tool joins an externally-supplied string onto a filesystem root and opens
    the result, and the check costs one comparison.
    """
    if root is None or not relative:
        return None

    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / str(relative)).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - unresolvable symlink chain
        return None

    if not resolved.is_relative_to(root_path):
        raise ValueError(
            f"path {relative!r} resolves outside the configured root "
            f"{root_path} (to {resolved}); refusing to read it"
        )
    return resolved if resolved.is_file() else None
