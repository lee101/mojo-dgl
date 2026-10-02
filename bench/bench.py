"""mojodgl against upstream dgl on the same arrays.

    pixi run bench

Reports the honest number, including the cases dgl wins: its kernels are
OpenMP-parallel C++ and this port is single-threaded scalar Mojo.
"""
import gc
import math
import os
import platform
import re
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojodgl as dgl  # noqa: E402
import mojodgl.function as fn  # noqa: E402
import mojodgl.ops as ops  # noqa: E402
import dgl as udgl  # noqa: E402
import dgl.function as ufn  # noqa: E402
import dgl.ops as uops  # noqa: E402
import torch  # noqa: E402

NODES = 200_000
EDGES = 1_000_000
DIM = 32

_CACHE = {}


def data():
    """One graph and its features, shared by every case."""
    if "main" in _CACHE:
        return _CACHE["main"]
    rng = np.random.default_rng(0)
    u = np.ascontiguousarray(rng.integers(0, NODES, size=EDGES))
    v = np.ascontiguousarray(rng.integers(0, NODES, size=EDGES))
    x = np.ascontiguousarray(rng.normal(size=(NODES, DIM)))
    e = np.ascontiguousarray(rng.normal(size=(EDGES, DIM)))
    z = np.ascontiguousarray(rng.normal(size=(EDGES, 1)))
    zh = np.ascontiguousarray(rng.normal(size=(EDGES, 8, 1)))
    g = dgl.graph((u, v))
    g.ndata["h"] = x
    g.edata["e"] = e
    ug = udgl.graph((torch.as_tensor(u), torch.as_tensor(v)))
    ug.ndata["h"] = torch.as_tensor(x)
    ug.edata["e"] = torch.as_tensor(e)
    _CACHE["main"] = (g, ug, x, e, z, zh)
    return _CACHE["main"]


def segment_data():
    if "seg" in _CACHE:
        return _CACHE["seg"]
    rng = np.random.default_rng(11)
    cuts = np.sort(rng.choice(EDGES, size=199_999, replace=False))
    seglen = np.diff(np.concatenate([[0], cuts, [EDGES]])).astype(np.int64)
    sv = np.ascontiguousarray(rng.normal(size=(EDGES, DIM)))
    _CACHE["seg"] = (seglen, sv)
    return _CACHE["seg"]


def mm_data():
    if "mm" in _CACHE:
        return _CACHE["mm"]
    rng = np.random.default_rng(12)
    a = np.ascontiguousarray(rng.normal(size=(EDGES, 16)))
    b = np.ascontiguousarray(rng.normal(size=(64, 16, 16)))
    idx = np.ascontiguousarray(rng.integers(0, 64, size=EDGES))
    _CACHE["mm"] = (a, b, idx)
    return _CACHE["mm"]


def timeit(call, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - t0)
    return best


def t(a):
    return torch.as_tensor(a)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("u_mul_e_sum (1M edges, 32d)")
def _():
    g, ug, x, e, *_ = data()
    return (lambda: ops.u_mul_e_sum(g, x, e),
            lambda: uops.u_mul_e_sum(ug, t(x), t(e)))


@case("u_add_e_mean (1M edges, 32d)")
def _():
    g, ug, x, e, *_ = data()
    return (lambda: ops.u_add_e_mean(g, x, e),
            lambda: uops.u_add_e_mean(ug, t(x), t(e)))


@case("u_div_e_min (1M edges, 32d)")
def _():
    g, ug, x, e, *_ = data()
    x = x + 4.0
    e = e + 4.0
    return (lambda: ops.u_div_e_min(g, x, e),
            lambda: uops.u_div_e_min(ug, t(x), t(e)))


@case("copy_u_max (1M edges, 32d)")
def _():
    g, ug, x, *_ = data()
    return (lambda: ops.copy_u_max(g, x), lambda: uops.copy_u_max(ug, t(x)))


@case("copy_e_mean (1M edges, 32d)")
def _():
    g, ug, _, e, *_ = data()
    return (lambda: ops.copy_e_mean(g, e), lambda: uops.copy_e_mean(ug, t(e)))


@case("u_dot_e (SDDMM, 1M edges, 32d)")
def _():
    g, ug, x, e, *_ = data()
    return (lambda: ops.u_dot_e(g, x, e), lambda: uops.u_dot_e(ug, t(x), t(e)))


@case("u_mul_v (SDDMM, 1M edges, 32d)")
def _():
    g, ug, x, *_ = data()
    return (lambda: ops.u_mul_v(g, x, x), lambda: uops.u_mul_v(ug, t(x), t(x)))


@case("edge_softmax norm_by=dst (1M edges)")
def _():
    g, ug, _, _, z, *_ = data()
    return (lambda: ops.edge_softmax(g, z), lambda: uops.edge_softmax(ug, t(z)))


@case("edge_softmax norm_by=src (1M edges)")
def _():
    g, ug, _, _, z, *_ = data()
    return (lambda: ops.edge_softmax(g, z, norm_by="src"),
            lambda: uops.edge_softmax(ug, t(z), norm_by="src"))


@case("edge_softmax 8 heads (1M edges, (E, 8, 1))")
def _():
    g, ug, _, _, _, zh = data()
    return (lambda: ops.edge_softmax(g, zh),
            lambda: uops.edge_softmax(ug, t(zh)))

@case("segment_reduce sum (1M rows, 32d, 200k segments)")
def _():
    seglen, sv = segment_data()
    return (lambda: ops.segment_reduce(seglen, sv, "sum"),
            lambda: uops.segment_reduce(torch.as_tensor(seglen), t(sv), "sum"))


@case("segment_reduce max (1M rows, 32d, 200k segments)")
def _():
    seglen, sv = segment_data()
    return (lambda: ops.segment_reduce(seglen, sv, "max"),
            lambda: uops.segment_reduce(torch.as_tensor(seglen), t(sv), "max"))


@case("segment_softmax (1M rows, 32d, 200k segments)")
def _():
    seglen, sv = segment_data()
    return (lambda: ops.segment_softmax(seglen, sv),
            lambda: uops.segment_softmax(torch.as_tensor(seglen), t(sv)))


@case("gather_mm (1M rows, 16x16, 64 blocks)")
def _():
    a, b, idx = mm_data()
    return (lambda: ops.gather_mm(a, b, idx_b=idx),
            lambda: uops.gather_mm(t(a), t(b), idx_b=torch.as_tensor(idx)))


@case("update_all u_mul_e + sum (1M edges, 32d)")
def _():
    g, ug, *_ = data()

    def ours():
        g.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "z"))
        return g.ndata["z"]

    def theirs():
        ug.update_all(ufn.u_mul_e("h", "e", "m"), ufn.sum("m", "z"))
        return ug.ndata["z"]

    return ours, theirs


@case("update_all copy_u + max (1M edges, 32d)")
def _():
    g, ug, *_ = data()

    def ours():
        g.update_all(fn.copy_u("h", "m"), fn.max("m", "z"))
        return g.ndata["z"]

    def theirs():
        ug.update_all(ufn.copy_u("h", "m"), ufn.max("m", "z"))
        return ug.ndata["z"]

    return ours, theirs


def machine() -> str:
    try:
        text = open("/proc/cpuinfo").read()
        name = re.search(r"model name\s*:\s*(.+)", text).group(1).strip()
        cores = len(re.findall(r"^processor", text, re.M))
    except (OSError, AttributeError):
        name, cores = platform.machine(), os.cpu_count() or 0
    return "{}, {} cores, Linux x86_64, torch {}".format(
        name, cores, torch.__version__
    )


def main() -> None:
    print("mojo-dgl vs upstream dgl")
    print("machine: {}".format(machine()))
    print()
    print("| case | mojo-dgl | dgl | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, build in CASES:
        ours, theirs = build()
        ours()  # warm the library load out of the measurement
        a = timeit(ours)
        b = timeit(theirs)
        print("| {} | {:.1f} ms | {:.1f} ms | {:.2f}x |".format(
            name, a * 1e3, b * 1e3, b / a
        ), flush=True)
        gc.collect()


if __name__ == "__main__":
    if "--locked" in sys.argv:
        # The caller already holds /tmp/mojo-bench.lock; do not queue again.
        main()
    else:
        raise SystemExit(
            "run this through `pixi run bench`, which takes the machine-wide "
            "lock so concurrent factory jobs cannot distort the numbers"
        )
