"""Make torch able to see the GPU on a NixOS-style driver layout.

Written after a wrong conclusion. Running the venv's python directly reports
``torch.cuda.is_available() == False`` on this machine, and that was reported as
"there is no usable GPU" -- a claim about hardware inferred from a fact about
environment. ``nvidia-smi`` works; the RTX 3060 is present and the driver is
configured at the NixOS host level.

The cause is the ordinary NixOS/manylinux mismatch. pip and uv install torch as
a manylinux wheel that resolves ``libcuda.so.1`` through the standard loader
path, while NixOS keeps the driver in ``/run/opengl-driver/lib``. Adding that
directory to ``LD_LIBRARY_PATH`` before the process starts makes CUDA available
with no other change.

**Timing matters, and this was measured rather than assumed.** The dynamic
loader caches its search path when the process starts, so setting
``os.environ["LD_LIBRARY_PATH"]`` before importing torch does *not* help the
process doing the setting -- verified: it still reports False. Two things do
work:

* ``ctypes.CDLL(".../libcuda.so.1", RTLD_GLOBAL)`` before torch initializes
  CUDA. An absolute path bypasses the search path entirely, and loading it
  globally means torch's own ``dlopen("libcuda.so.1")`` finds it already
  resident. This fixes the *current* process, so it is what runs at import.
* Exporting ``LD_LIBRARY_PATH`` in the environment before launch. This is what
  fixes *child* processes, so it is also set, for workers and subprocesses.

Both are applied at import, because the two cover different processes.
"""

from __future__ import annotations

import ctypes
import os

__all__ = ["DRIVER_LIB_DIR", "ensure_driver_path", "preload_libcuda", "cuda_diagnosis"]

#: Where NixOS exposes the graphics/compute driver, including ``libcuda.so.1``.
DRIVER_LIB_DIR = "/run/opengl-driver/lib"


def ensure_driver_path(directory: str = DRIVER_LIB_DIR) -> bool:
    """Prepend the driver directory to ``LD_LIBRARY_PATH`` if it exists.

    Returns True when the path is present afterwards. Idempotent, so importing
    this module from several places cannot grow the variable without bound.
    """
    if not os.path.isdir(directory):
        return False
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [p for p in current.split(os.pathsep) if p]
    if directory not in entries:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([directory, *entries])
    return True


def preload_libcuda(directory: str = DRIVER_LIB_DIR) -> bool:
    """Load ``libcuda.so.1`` by absolute path so the current process can use it.

    Must run before torch initializes CUDA. Loading with ``RTLD_GLOBAL`` puts
    the library in the global symbol namespace, so torch's later
    ``dlopen("libcuda.so.1")`` resolves against the already-resident copy
    instead of searching a path that does not contain it.

    Returns True if the driver is loaded and usable. False is not an error: it
    is the normal answer on a machine with no NVIDIA driver.
    """
    path = os.path.join(directory, "libcuda.so.1")
    if not os.path.exists(path):
        return False
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def cuda_diagnosis(*, cuda_available: bool, nvidia_smi_ok: bool) -> str:
    """Explain a CUDA state, naming the fix rather than the symptom.

    The case worth getting right is `nvidia-smi` seeing a card while torch does
    not: reporting that as "no GPU" is what turned an environment problem into a
    wrong conclusion about what this project could attempt.
    """
    if cuda_available:
        return "CUDA available: torch can reach the GPU."
    if nvidia_smi_ok:
        return (
            "nvidia-smi sees a GPU but torch cannot use it. This is almost "
            "certainly the loader path, not the hardware: pip/uv install torch as "
            "a manylinux wheel that looks for libcuda.so.1 on the standard path, "
            f"while the driver lives in {DRIVER_LIB_DIR}. Either import "
            "enhancement_eval.gpu before torch, or start the process with\n"
            f"    LD_LIBRARY_PATH={DRIVER_LIB_DIR}:$LD_LIBRARY_PATH\n"
            "Exporting that variable inside an already-running interpreter does "
            "not help it -- the loader caches its search path at startup."
        )
    return "No NVIDIA GPU detected on this machine (nvidia-smi absent or failing)."


# Both are applied at import: the preload fixes this process, the environment
# variable fixes anything spawned from it.
ensure_driver_path()
preload_libcuda()
