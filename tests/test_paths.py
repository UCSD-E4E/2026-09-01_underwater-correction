"""Resolving NAS-relative paths from the manifest.

`Image.path` and `DiveSlate.path` are share-relative (`2024.06.20.REEF/...`),
rooted at the worker's `e4e_nas.raw_root_path` -- on this box,
`~/mnt/fishsense_data/REEF/data`. Resolving them directly beats copying
thousands of `.ORF` files into a checksum-named cache, so both are supported:
the cache when it exists, the NAS root otherwise.

The traversal guard is not paranoia about the database. It is that a path
joined from *any* external source and then opened is the one place this
read-only tool could be talked into reading something outside the share, and
the check costs one comparison.
"""

from pathlib import Path

import pytest

from enhancement_eval.paths import resolve_under_root


def test_resolves_a_share_relative_path(tmp_path):
    target = tmp_path / "2024.06.20.REEF" / "photos" / "P8010001.ORF"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert resolve_under_root(tmp_path, "2024.06.20.REEF/photos/P8010001.ORF") == target


def test_handles_spaces_and_hashes_in_names(tmp_path):
    """Real slate templates are called things like `dive slate#6.pdf`."""
    target = tmp_path / "old slates" / "dive slate#6.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert resolve_under_root(tmp_path, "old slates/dive slate#6.pdf") == target


def test_missing_file_returns_none_rather_than_raising(tmp_path):
    """A dive whose raws were never synced is an ordinary outcome on a
    partially-populated mount, not an error that should stop a sweep."""
    assert resolve_under_root(tmp_path, "nope/missing.ORF") is None


def test_refuses_to_escape_the_root(tmp_path):
    outside = tmp_path.parent / "secret.ORF"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="outside"):
        resolve_under_root(tmp_path / "root", f"../{outside.name}")


def test_refuses_an_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="outside|absolute"):
        resolve_under_root(tmp_path, "/etc/passwd")


def test_empty_path_is_none(tmp_path):
    assert resolve_under_root(tmp_path, "") is None
    assert resolve_under_root(tmp_path, None) is None


def test_no_root_configured_is_none(tmp_path):
    assert resolve_under_root(None, "anything.ORF") is None
