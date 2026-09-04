"""Make a repo-local `.pylib` importable.

The `bm3d` package is not in the shared fishsense-lite venv, and that venv
belongs to another repo. `uv pip install --python <venv> --target .pylib bm3d`
puts it here instead (drop the numpy/scipy it drags along -- the venv has
them, and a second copy ahead on the path shadows the real one). The tests
that need it skip if it is absent.
"""
import pathlib
import sys

_pylib = pathlib.Path(__file__).resolve().parent.parent / ".pylib"
if _pylib.is_dir() and str(_pylib) not in sys.path:
    sys.path.append(str(_pylib))
