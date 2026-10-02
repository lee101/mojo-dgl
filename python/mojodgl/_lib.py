"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

from .base import DGLError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJODGL_LIB") or os.path.join(ROOT, "dist", "libmojo-dgl.so")


class BuildError(RuntimeError):
    pass


def _source_dir() -> str:
    """The Mojo sources, whether this is an in-tree checkout or an install.

    In a checkout they sit at `<repo>/src`, which is what the pixi tasks build.
    A wheel ships only the Python package, so there is no `src` next to it; the
    Mojo sources have to be installed alongside for the lazy build to work. They
    are looked for beside the package first, then next to it, and this is what
    `pyproject.toml` puts on `tool.hatch.build.targets.wheel`.
    """
    override = os.environ.get("MOJODGL_SRC")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "mojo"),       # installed: python/mojodgl/mojo/
        os.path.join(ROOT, "src"),        # in-tree checkout
    ):
        if os.path.isfile(os.path.join(candidate, "capi.mojo")):
            return candidate
    raise BuildError(
        "the Mojo sources are not installed alongside the package; set "
        "MOJODGL_SRC=/path/to/src, which must contain capi.mojo"
    )

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
    "mdgl_edge_softmax_": ([I] * 10, None),
    "mdgl_gather_mm": ([I] * 7, None),
    "mdgl_workers": ([], I),
}

# src/ported.mojo
SRC_CODE, DST_CODE, EDGE_CODE = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_DOT = 0, 1, 2, 3, 4
OP_COPY_LHS, OP_COPY_RHS = 5, 6
REDUCE_SUM, REDUCE_MAX, REDUCE_MIN, REDUCE_MEAN = 0, 1, 2, 3

BINARY_OPS = {"add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV, "dot": OP_DOT}
REDUCERS = {"sum": REDUCE_SUM, "max": REDUCE_MAX, "min": REDUCE_MIN, "mean": REDUCE_MEAN}
TARGETS = {"u": SRC_CODE, "v": DST_CODE, "e": EDGE_CODE}


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
    """Compile the Mojo sources into `dist/libmojo-dgl.so` if it is stale."""
    src = _source_dir()
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(src)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB) or ".", exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", src,
        os.path.join(src, "capi.mojo"), "-o", LIB,
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
    """A C-contiguous float64 view of `a`, copying only when it has to.

    The kernels are Float64, so this is the one place a dtype becomes an FFI
    assumption. A conversion that would silently change a value is rejected
    rather than performed: complex data (the imaginary part is dropped), and
    every integer dtype except int64 whose values float64's 53-bit mantissa
    cannot represent exactly. int64 features are rounded to float64, which is
    the documented float64-only limit rather than a narrowing surprise; every
    other conversion is value-preserving to float64's precision, which is the
    same widening upstream gets when torch promotes its own tensors.
    """
    a = np.asanyarray(a)
    if np.iscomplexobj(a):
        raise DGLError(
            "this port's kernels are float64 and complex input is not supported"
        )
    if np.issubdtype(a.dtype, np.integer) and a.dtype != np.int64 and a.size:
        # Every integer dtype except int64 is checked element by element, because
        # the range alone does not catch a value in the middle that float64's
        # 53-bit mantissa cannot represent. int64 itself is not checked: this
        # port is float64 end to end, so an int64 feature is rounded by design,
        # which is the documented float64-only limit rather than a surprise.
        # Note this covers uint64 too, which is also 8 bytes wide.
        with np.errstate(invalid="ignore"):
            # A value outside the dtype's range casts to the cast's "undefined"
            # result with a warning; the round trip below is what decides, so
            # the warning is not useful here.
            back = a.astype(np.float64).astype(a.dtype)
        if not bool(np.array_equal(back, a)):
            raise DGLError(
                "{} values do not survive the conversion to float64 exactly".format(
                    a.dtype
                )
            )
    return np.array(a, dtype=np.float64, order="C", copy=True) if copy else \
        np.ascontiguousarray(a, dtype=np.float64)


def i64(a) -> np.ndarray:
    """A C-contiguous int64 view of `a`, for node ids and segment lengths.

    Rejects non-integral floats rather than truncating them into node ids that
    silently point at the wrong row.
    """
    a = np.asanyarray(a)
    if np.issubdtype(a.dtype, np.floating) and a.size:
        if not bool(np.all(a == np.floor(a))):
            raise DGLError(
                "node ids and segment lengths must be whole numbers, got {}".format(
                    a.dtype
                )
            )
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
