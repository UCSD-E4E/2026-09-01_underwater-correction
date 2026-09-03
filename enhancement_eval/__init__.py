"""Offline evaluation of underwater image enhancement against the label corpus.

Read-only against production. See README.md for the reasoning and `cli.py` for
the entry point.

----------------------------------------------------------------------------
Why this module imports torch before anything else
----------------------------------------------------------------------------
`torch` must initialize its threading runtime before `rawpy` loads LibRaw's,
or CPU inference segfaults -- reliably, with no Python traceback, deep inside
`torch.nn.Linear.forward`.

This was not theoretical. It took down the end-to-end test on every run, and it
is exactly the shape of bug that would otherwise appear halfway through an
overnight sweep over several hundred frames: no error, no partial results, just
a core dump. Bisected to this: with `import torch` at process start both
orderings succeed, and without it both crash, whether the raw is decoded before
the model is loaded or after.

The import is eager and unconditional-if-available rather than tucked inside
the module that runs models, because the ordering is a property of the
*process*, not of any one call site -- `stages.py` imports rawpy lazily inside
functions, so by the time anything realises it needs torch, LibRaw has already
loaded. Importing here means anything that touches this package gets the order
right without having to know the rule.

It is best-effort: the pure-logic modules (`contract`, `metrics`, `sites`,
`scoring`) have no torch dependency and stay importable without it.

`gpu` is imported first for a second, unrelated ordering rule: it preloads the
NVIDIA driver by absolute path, which only has any effect if it happens before
torch initializes CUDA. Without it torch on this machine silently falls back to
CPU -- which is how a working RTX 3060 got reported as "no usable GPU".
"""

from . import gpu as _gpu  # noqa: F401  (side effect: make the driver findable)

try:  # pragma: no cover - environment-dependent by nature
    import torch as _torch  # noqa: F401  (imported for its side effect: see above)
except ImportError:  # pragma: no cover
    _torch = None
