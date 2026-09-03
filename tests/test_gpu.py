"""Torch must be able to see the GPU, when there is one.

This exists because of a wrong conclusion. Running the venv's python directly
reports `torch.cuda.is_available() == False` on this machine, and that was
reported as "no GPU available" -- a claim about hardware drawn from a fact
about environment. `nvidia-smi` works fine; the driver is configured at the
NixOS host level.

The cause is the standard NixOS/manylinux mismatch: pip and uv install torch as
a manylinux wheel that resolves `libcuda.so.1` through the normal loader path,
while NixOS keeps the driver in `/run/opengl-driver/lib`. Adding that directory
to `LD_LIBRARY_PATH` makes CUDA available immediately.

The test is skipped where there is genuinely no GPU, so it does not fail in CI
or on a laptop without one. What it will not let happen again is a *silent*
fallback to CPU on a machine that has a working card.
"""

import os
import shutil
import subprocess

import pytest

from enhancement_eval.gpu import (
    DRIVER_LIB_DIR,
    cuda_diagnosis,
    ensure_driver_path,
    preload_libcuda,
)


def _has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def test_driver_path_is_added_when_it_exists():
    if not os.path.isdir(DRIVER_LIB_DIR):
        pytest.skip(f"{DRIVER_LIB_DIR} not present; not a NixOS-style driver layout")
    before = os.environ.get("LD_LIBRARY_PATH", "")
    try:
        os.environ["LD_LIBRARY_PATH"] = ""
        assert ensure_driver_path() is True
        assert DRIVER_LIB_DIR in os.environ["LD_LIBRARY_PATH"]
    finally:
        os.environ["LD_LIBRARY_PATH"] = before


def test_adding_the_path_twice_does_not_duplicate_it():
    if not os.path.isdir(DRIVER_LIB_DIR):
        pytest.skip("no driver dir")
    before = os.environ.get("LD_LIBRARY_PATH", "")
    try:
        os.environ["LD_LIBRARY_PATH"] = ""
        ensure_driver_path()
        ensure_driver_path()
        assert os.environ["LD_LIBRARY_PATH"].count(DRIVER_LIB_DIR) == 1
    finally:
        os.environ["LD_LIBRARY_PATH"] = before


def test_diagnosis_explains_a_visible_gpu_that_torch_cannot_use():
    """The specific failure that wasted a conclusion: nvidia-smi sees a card,
    torch does not. The message must name the fix rather than report 'no GPU'."""
    text = cuda_diagnosis(cuda_available=False, nvidia_smi_ok=True)
    assert "LD_LIBRARY_PATH" in text
    assert DRIVER_LIB_DIR in text
    assert "no gpu" not in text.lower()


def test_diagnosis_of_a_machine_with_no_card_says_so_plainly():
    text = cuda_diagnosis(cuda_available=False, nvidia_smi_ok=False)
    assert "no nvidia gpu" in text.lower()


def test_diagnosis_when_everything_works_is_short():
    assert "available" in cuda_diagnosis(cuda_available=True, nvidia_smi_ok=True).lower()


def test_preload_reports_whether_the_driver_loaded():
    assert preload_libcuda() is os.path.exists(os.path.join(DRIVER_LIB_DIR, "libcuda.so.1"))


def test_preload_of_a_missing_directory_is_false_not_an_exception():
    assert preload_libcuda("/nonexistent/driver/dir") is False


@pytest.mark.integration
def test_importing_the_package_lets_torch_reach_the_gpu():
    """The claim that matters, tested the only way it can be: in a fresh
    process with the environment stripped.

    Setting os.environ["LD_LIBRARY_PATH"] before importing torch does NOT work
    -- measured, it still reports False, because the loader caches its search
    path at startup. The ctypes preload does work. This test would have caught
    the wrong conclusion that a working GPU was unusable.
    """
    if not _has_nvidia_gpu():
        pytest.skip("no NVIDIA GPU on this machine")
    import sys

    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    out = subprocess.run(
        [sys.executable, "-c", "import enhancement_eval, torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=300, env=env, cwd=os.getcwd(),
    )
    assert out.stdout.strip().endswith("True"), (
        f"importing the package did not make CUDA reachable:\n{out.stdout}\n{out.stderr[-600:]}"
    )
