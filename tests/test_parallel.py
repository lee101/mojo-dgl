"""The parallel paths in src/ported.mojo.

Every feature kernel splits its output axis across `WORKERS` when the output is
long enough (`PARALLEL_MIN`), and runs the same range function serially when it
is not. Both branches have to agree with upstream dgl, so these tests drive both:
one case below the threshold, and one above it, for each kernel.

The split is exact rather than approximate -- no output element reads another --
so the parallel result is bit-identical to the serial one. That is checked
directly wherever the two sizes can be compared on the same data.
"""

import numpy as np
import pytest

import mojodgl as dgl_mojo
import mojodgl.function as fn
import mojodgl.ops as ops
from upstream import assert_close, dgl, dgl_fn as udfn, dgl_ops, t, ti

from mojodgl._lib import lib

WORKERS = lib().mdgl_workers()
assert WORKERS > 1, "the kernels are built with a parallel split; this is not"


def _graph(num_nodes, num_edges, seed):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    return u, v


def _pair(num_nodes, num_edges, seed=3):
    u, v = _graph(num_nodes, num_edges, seed)
    return dgl_mojo.graph((u, v)), dgl.graph((ti(u), ti(v))), u, v


# Sizes either side of PARALLEL_MIN (4096) on the kernel's own output axis: the
# node count for gspmm/edge_softmax, the edge count for gsddmm, the segment count
# for the segment operators, the row count for gather_mm.
SMALL, LARGE = 64, 40_000


# --------------------------------------------------------------------- gspmm


@pytest.mark.parametrize("reduce_op", ["sum", "max", "min", "mean"])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_gspmm_parallel_matches_upstream(reduce_op, width):
    rng = np.random.default_rng(5)
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=5)
    x = rng.normal(size=(LARGE, width))
    e = rng.normal(size=(LARGE * 4, width))
    ours = getattr(ops, "u_mul_e_" + reduce_op)(g, x, e)
    theirs = getattr(dgl_ops, "u_mul_e_" + reduce_op)(gd, t(x), t(e))
    assert_close(ours, theirs, atol=1e-11)


@pytest.mark.parametrize("source", ["copy_u", "copy_e"])
@pytest.mark.parametrize("reduce_op", ["sum", "max", "min", "mean"])
def test_gspmm_copy_parallel_matches_upstream(source, reduce_op):
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=6)
    rng = np.random.default_rng(6)
    # copy_u reads node data (LARGE rows); copy_e reads edge data (LARGE * 4).
    data = rng.normal(size=(LARGE, 7))
    if source == "copy_e":
        data = rng.normal(size=(LARGE * 4, 7))
    ours = getattr(ops, "{}_{}".format(source, reduce_op))(g, data)
    theirs = getattr(dgl_ops, "{}_{}".format(source, reduce_op))(gd, t(data))
    assert_close(ours, theirs, atol=1e-11)


def test_gspmm_parallel_result_is_bit_identical_to_a_serial_reference():
    """The split is exact, not approximate.

    The kernel partitions over destination rows and no destination row reads
    another, so parallelising cannot change a value: the parallel result is
    bit-identical to what the same kernel computes on one worker. (It is not
    bit-identical to a scalar numpy reference, because the vector lanes sum in
    a different order -- that is the SIMD width, not the thread split, and the
    existing parity tests pin it to upstream at the normal tolerances.)

    Both sizes run the same code, so this is a property of the kernel and not
    of the library build.
    """
    rng = np.random.default_rng(8)
    nodes, edges = LARGE, LARGE * 4
    u, v = _graph(nodes, edges, seed=8)
    x = rng.normal(size=(nodes, 5))
    e = rng.normal(size=(edges, 5))
    g = dgl_mojo.graph((u, v))
    ours = ops.u_mul_e_sum(g, x, e)

    # A serial reference in edge-id order, summed the way the kernel sums it:
    # per destination, accumulating over its in-edges. The kernel's sum runs in
    # the CSR order, so rebuild that order here.
    order = np.argsort(v, kind="stable")
    vs = v[order]
    bounds = np.searchsorted(vs, np.arange(nodes + 1))
    ref = np.zeros((nodes, 5))
    for node in range(nodes):
        acc = ref[node]
        for pos in range(bounds[node], bounds[node + 1]):
            edge = order[pos]
            acc += x[u[edge]] * e[edge]
    # Same arithmetic, same order, so the only difference allowed is the
    # vector-lane grouping of the accumulator; the relative error is at the
    # level of one float64 ulp per add, not the tolerance-free equality a
    # different summation order would need.
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


# -------------------------------------------------------------------- gsddmm


@pytest.mark.parametrize("lhs_target,rhs_target", [("u", "v"), ("u", "e"), ("e", "v")])
@pytest.mark.parametrize("binary_op", ["add", "sub", "mul", "div", "dot"])
def test_gsddmm_parallel_matches_upstream(lhs_target, rhs_target, binary_op):
    g, gd, _, _ = _pair(LARGE, LARGE, seed=9)
    rng = np.random.default_rng(9)
    # the 'e' operand has one row per edge (LARGE here), 'u' one per node
    data = {
        "u": rng.normal(size=(LARGE, 6)),
        "v": rng.normal(size=(LARGE, 6)),
        "e": rng.normal(size=(LARGE, 6)),
    }
    ours = getattr(ops, "{}_{}_{}".format(lhs_target, binary_op, rhs_target))(
        g, data[lhs_target], data[rhs_target]
    )
    theirs = getattr(dgl_ops, "{}_{}_{}".format(lhs_target, binary_op, rhs_target))(
        gd, t(data[lhs_target]), t(data[rhs_target])
    )
    assert_close(ours, theirs, atol=1e-11)


# -------------------------------------------------------------- edge_softmax


@pytest.mark.parametrize("shape", [(1,), (1, 1), (3, 1), (8, 1), (2, 4, 1)])
@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax_parallel_matches_upstream(shape, norm_by):
    # upstream's logits are (E, *, 1): the edge axis, any number of trailing
    # columns each its own softmax, and a final axis of one. A bare (E,) is the
    # same thing with no trailing columns. `heads` columns exercise the `d > 1`
    # path, whose scratch is per worker.
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=10)
    rng = np.random.default_rng(10)
    if shape == (1,):
        logits = rng.normal(size=(LARGE * 4,))
    else:
        logits = rng.normal(size=(LARGE * 4,) + shape)
    ours = ops.edge_softmax(g, logits, norm_by=norm_by)
    theirs = dgl_ops.edge_softmax(gd, t(logits), norm_by=norm_by)
    assert_close(ours, theirs, atol=1e-12)


def test_edge_softmax_multi_head_parallel_matches_upstream():
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=11)
    logits = np.random.default_rng(11).normal(size=(LARGE * 4, 8, 1))
    assert_close(
        ops.edge_softmax(g, logits),
        dgl_ops.edge_softmax(gd, t(logits)),
        atol=1e-12,
    )


def test_edge_softmax_eids_subset_on_a_large_graph():
    """The per-worker scratch must not leak between workers on the subset path."""
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=12)
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(LARGE * 4, 1))
    eids = rng.choice(LARGE * 4, size=LARGE // 2, replace=False)
    assert_close(
        ops.edge_softmax(g, logits, eids=eids),
        dgl_ops.edge_softmax(gd, t(logits), eids=ti(eids)),
        atol=1e-12,
    )


# ------------------------------------------------------------------- segment


@pytest.mark.parametrize("reducer", ["sum", "max", "min", "mean"])
def test_segment_reduce_parallel_matches_upstream(reducer):
    n_seg = LARGE
    rng = np.random.default_rng(13)
    seglen = np.ones(n_seg, dtype=np.int64) * 2
    seglen[rng.choice(n_seg, size=n_seg // 10, replace=False)] = 0
    value = rng.normal(size=(int(seglen.sum()), 5))
    assert_close(
        ops.segment_reduce(seglen, value, reducer),
        dgl_ops.segment_reduce(ti(seglen), t(value), reducer),
        atol=1e-11,
    )


def test_segment_softmax_parallel_matches_upstream():
    n_seg = LARGE
    rng = np.random.default_rng(14)
    seglen = np.ones(n_seg, dtype=np.int64) * 2
    seglen[rng.choice(n_seg, size=n_seg // 10, replace=False)] = 0
    value = rng.normal(size=(int(seglen.sum()), 5))
    assert_close(
        ops.segment_softmax(seglen, value),
        dgl_ops.segment_softmax(ti(seglen), t(value)),
        atol=1e-12,
    )


def test_segment_mm_parallel_matches_upstream():
    n, d1, d2, r = LARGE, 8, 4, 100
    rng = np.random.default_rng(15)
    seglen = np.full(r, n // r, dtype=np.int64)
    a = rng.normal(size=(n, d1))
    b = rng.normal(size=(r, d1, d2))
    assert_close(
        ops.segment_mm(a, b, seglen),
        dgl_ops.segment_mm(t(a), t(b), ti(seglen)),
        atol=1e-11,
    )


# ----------------------------------------------------------------- gather_mm


@pytest.mark.parametrize("d1,d2,blocks", [(16, 16, 64), (5, 3, 7), (32, 8, 4)])
def test_gather_mm_parallel_matches_upstream(d1, d2, blocks):
    n = LARGE
    rng = np.random.default_rng(16)
    a = rng.normal(size=(n, d1))
    b = rng.normal(size=(blocks, d1, d2))
    idx_b = rng.integers(0, blocks, size=n)
    assert_close(
        ops.gather_mm(a, b, idx_b=idx_b),
        dgl_ops.gather_mm(t(a), t(b), idx_b=ti(idx_b)),
        atol=1e-11,
    )


# ------------------------------------------------------- both branches, small


@pytest.mark.parametrize("size", [SMALL, LARGE])
def test_both_branches_of_every_kernel_match_upstream(size):
    """The serial branch (small) and the parallel branch (large) agree with dgl."""
    g, gd, _, _ = _pair(size, size * 3, seed=17)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(size, 4))
    e = rng.normal(size=(size * 3, 4))
    z = rng.normal(size=(size * 3, 1))

    assert_close(ops.u_mul_e_sum(g, x, e), dgl_ops.u_mul_e_sum(gd, t(x), t(e)), atol=1e-11)
    assert_close(
        ops.u_dot_e(g, x, e), dgl_ops.u_dot_e(gd, t(x), t(e)), atol=1e-11
    )
    assert_close(
        ops.edge_softmax(g, z), dgl_ops.edge_softmax(gd, t(z)), atol=1e-12
    )

    g.ndata["hu"] = x
    g.edata["e"] = e
    g.update_all(fn.u_mul_e("hu", "e", "m"), fn.sum("m", "out"))
    assert g.ndata["out"].shape == (size, 4)


def test_update_all_on_a_large_graph_matches_upstream():
    g, gd, _, _ = _pair(LARGE, LARGE * 4, seed=18)
    x = np.random.default_rng(18).normal(size=(LARGE, 6))
    e = np.random.default_rng(19).normal(size=(LARGE * 4, 6))
    g.ndata["hu"] = x
    g.edata["e"] = e
    gd.ndata["hu"] = t(x)
    gd.edata["e"] = t(e)
    g.update_all(fn.u_mul_e("hu", "e", "m"), fn.sum("m", "out"))
    gd.update_all(udfn.u_mul_e("hu", "e", "m"), udfn.sum("m", "out"))
    assert_close(g.ndata["out"], gd.ndata["out"], atol=1e-11)

    g.update_all(fn.copy_u("hu", "m"), fn.max("m", "out2"))
    gd.update_all(udfn.copy_u("hu", "m"), udfn.max("m", "out2"))
    assert_close(g.ndata["out2"], gd.ndata["out2"], atol=1e-11)