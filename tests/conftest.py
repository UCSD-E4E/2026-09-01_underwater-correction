"""Shared fixtures.

The `.ORF` fixture is borrowed from the data-worker's test suite rather than
copied: it is 15 MB, it is already version-controlled there, and a second copy
would drift. Its path is resolved from the repo root so this project can sit
outside the monorepo.
"""

import os
from pathlib import Path

import pytest

#: Overridable so the harness can be run against the repo in another location.
FISHSENSE_LITE = Path(
    os.environ.get(
        "FISHSENSE_LITE_ROOT",
        Path.home() / "Repos/school/e4e/fishsense/fishsense-lite",
    )
)

ORF_FIXTURE = (
    FISHSENSE_LITE
    / "services/fishsense-data-processing-workflow-worker/tests/fixtures/stage2_sample.ORF"
)


@pytest.fixture(scope="session")
def orf_path() -> Path:
    if not ORF_FIXTURE.exists():
        pytest.skip(f"no .ORF fixture at {ORF_FIXTURE}")
    return ORF_FIXTURE


@pytest.fixture(scope="session")
def orf_bytes(orf_path: Path) -> bytes:
    return orf_path.read_bytes()
