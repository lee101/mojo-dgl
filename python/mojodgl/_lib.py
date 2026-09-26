"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJODGL_LIB") or os.path.join(ROOT, "dist", "libmojo-dgl.so")

I = ctypes.c_int64

# name -> (argtypes, restype). Every buffer crosses as an Int address; see
# src/capi.mojo for why.
_SIGNATURES = {
    "mdgl_cumsum": ([I, I, I], None),
    "mdgl_replace_inf_with_zero": ([I, I, I], None),
    "mdgl_clamp_i64": ([I, I, I, I, I], None),
    "mdgl_gsddmm": ([I] * 12, None),
    "mdgl_gspmm": ([I] * 13, None),
    "mdgl_segment_reduce": ([I] * 7, None),
    "mdgl_segment_softmax": ([I] * 7, None),
    "mdgl_segment_mm": ([I] * 8, None),
    "mdgl_edge_softmax_": ([I] * 8, None),
    "mdgl_gather_mm": ([I] * 7, None),
}

# src/ported.mojo
SRC_CODE, DST_CODE, EDGE_CODE = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_DOT = 0, 1, 2, 3, 4
OP_COPY_LHS, OP_COPY_RHS = 5, 6
REDUCE_SUM, REDUCE_MAX, REDUCE_MIN, REDUCE_MEAN = 0, 1, 2, 3

BINARY_OPS = {"add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV, "dot": OP_DOT}
REDUCERS = {"sum": REDUCE_SUM, "max": REDUCE_MAX, "min": REDUCE_MIN, "mean": REDUCE_MEAN}
TARGETS = {"u": SRC_CODE, "v": DST_CODE, "e": EDGE_CODE}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJODGL_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJODGL_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-dgl.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    return np.array(a, dtype=np.float64, order="C", copy=True) if copy else \
        np.ascontiguousarray(a, dtype=np.float64)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def empty(n: int, d: int = 1) -> np.ndarray:
    return np.empty((n, d) if d != 1 else (n,), dtype=np.float64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojodgl._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
