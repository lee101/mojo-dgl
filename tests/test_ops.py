"""Every dgl.ops entry point, checked against upstream dgl on the same data."""

import itertools

import importlib

import numpy as np
import pytest

import mojodgl as dgl_mojo
import mojodgl.ops as ops
import mojodgl.ops.sddmm as sddmm_mod
from upstream import assert_close, dgl, dgl_ops, random_graph, t, ti

# `mojodgl.ops` re-exports the `gather_mm` function under the submodule's own
# name, so the module object has to come from the import system directly.
gather_mm_mod = importlib.import_module("mojodgl.ops.gather_mm")

U, V, N = random_graph(num_nodes=64, num_edges=512, seed=0)
G_MOJO = dgl_mojo.graph((U, V))
G_DGL = dgl.graph((ti(U), ti(V)))

rng = np.random.default_rng(7)
X = rng.normal(size=(N, 8))
E = rng.normal(size=(len(U), 8))
Y1 = rng.normal(size=(N, 1))


@pytest.mark.parametrize("binary_op", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("reduce_op", ["sum", "max", "min", "mean"])
def test_gspmm_families(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    ours = getattr(ops, name)(G_MOJO, X, E)
    theirs = getattr(dgl_ops, name)(G_DGL, t(X), t(E))
    assert_close(ours, theirs, atol=1e-11, label=name)


@pytest.mark.parametrize("source", ["copy_u", "copy_e"])
@pytest.mark.parametrize("reduce_op", ["sum", "max", "min", "mean"])
def test_gspmm_copy_families(source, reduce_op):
    name = "{}_{}".format(source, reduce_op)
    data = X if source == "copy_u" else E
    ours = getattr(ops, name)(G_MOJO, data)
    theirs = getattr(dgl_ops, name)(G_DGL, t(data))
    assert_close(ours, theirs, atol=1e-11, label=name)


def test_gspmm_broadcasts_a_scalar_column():
    ours = ops.u_add_e_sum(G_MOJO, Y1, E)
    theirs = dgl_ops.u_add_e_sum(G_DGL, t(Y1), t(E))
    assert_close(ours, theirs, atol=1e-11)


@pytest.mark.parametrize("lhs_target,rhs_target", list(itertools.permutations("uve", 2)))
@pytest.mark.parametrize("binary_op", ["add", "sub", "mul", "div", "dot"])
def test_gsddmm_families(lhs_target, rhs_target, binary_op):
    name = "{}_{}_{}".format(lhs_target, binary_op, rhs_target)
    data = {"u": X, "v": X, "e": E}
    ours = getattr(ops, name)(G_MOJO, data[lhs_target], data[rhs_target])
    theirs = getattr(dgl_ops, name)(G_DGL, t(data[lhs_target]), t(data[rhs_target]))
    assert_close(ours, theirs, atol=1e-10, label=name)


def test_gsddmm_copy_lhs_and_rhs():
    assert_close(
        ops.gsddmm(G_MOJO, "copy_lhs", X, None),
        dgl_ops.gsddmm(G_DGL, "copy_lhs", t(X), None),
    )
    assert_close(
        ops.gsddmm(G_MOJO, "copy_rhs", None, Y1),
        dgl_ops.gsddmm(G_DGL, "copy_rhs", None, t(Y1)),
    )
    assert_close(ops.copy_u(G_MOJO, X), dgl_ops.copy_u(G_DGL, t(X)))
    assert_close(ops.copy_v(G_MOJO, X), dgl_ops.copy_v(G_DGL, t(X)))
    assert_close(ops.copy_e(G_MOJO, E), dgl_ops.copy_e(G_DGL, t(E)))


def test_reshape_lhs_rhs_matches_upstream():
    a, b = sddmm_mod.reshape_lhs_rhs(np.zeros((4, 3, 1)), np.zeros((5, 3, 3, 4)))
    assert a.shape == (4, 1, 3, 1)
    assert b.shape == (5, 3, 3, 4)

@pytest.mark.parametrize("reducer", ["sum", "max", "min", "mean"])
def test_segment_reduce(reducer):
    seglen = np.array([3, 0, 5, 4, 2])
    value = np.arange(14 * 3, dtype=np.float64).reshape(14, 3)
    ours = ops.segment_reduce(seglen, value, reducer)
    theirs = dgl_ops.segment_reduce(ti(seglen), t(value), reducer)
    assert_close(ours, theirs, atol=1e-11, label=reducer)


def test_segment_reduce_rejects_unknown_reducer():
    with pytest.raises(dgl_mojo.DGLError):
        ops.segment_reduce(np.array([4]), np.ones((4, 1)), "median")


def test_segment_softmax():
    seglen = np.array([1, 0, 5, 4])
    value = np.arange(10, dtype=np.float64).reshape(10, 1)
    ours = ops.segment_softmax(seglen, value)
    theirs = dgl_ops.segment_softmax(ti(seglen), t(value))
    assert_close(ours, theirs, atol=1e-12)


def test_segment_mm():
    seglen = np.array([10, 5, 0, 3])
    a = np.arange(18 * 4, dtype=np.float64).reshape(18, 4)
    b = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)
    ours = ops.segment_mm(a, b, seglen)
    theirs = dgl_ops.segment_mm(t(a), t(b), ti(seglen))
    assert_close(ours, theirs, atol=1e-10)


def test_gather_mm_small_path():
    a = np.arange(6 * 4, dtype=np.float64).reshape(6, 4)
    b = np.arange(5 * 4 * 3, dtype=np.float64).reshape(5, 4, 3)
    idx = np.array([0, 3, 1, 1, 4, 2], dtype=np.int64)
    ours = ops.gather_mm(a, b, idx_b=idx)
    theirs = dgl_ops.gather_mm(t(a), t(b), idx_b=ti(idx))
    assert_close(ours, theirs, atol=1e-10)


def test_segment_reduce_rejects_lengths_that_do_not_add_up():
    with pytest.raises(dgl_mojo.DGLError):
        ops.segment_reduce(np.array([3, 3]), np.ones((4, 2)), "sum")


def test_segment_reduce_rejects_negative_lengths():
    with pytest.raises(dgl_mojo.DGLError):
        ops.segment_reduce(np.array([6, -2]), np.ones((4, 2)), "sum")



@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax(norm_by):
    logits = rng.normal(size=(len(U), 1))
    ours = ops.edge_softmax(G_MOJO, logits, norm_by=norm_by)
    theirs = dgl_ops.edge_softmax(G_DGL, t(logits), norm_by=norm_by)
    assert_close(ours, theirs, atol=1e-12, label=norm_by)


def test_edge_softmax_on_a_subset_of_edges():
    logits = rng.normal(size=(len(U), 1))
    eids = np.array([0, 5, 17, 30, 100, 511], dtype=np.int64)
    ours = ops.edge_softmax(G_MOJO, logits, eids=eids)
    theirs = dgl_ops.edge_softmax(G_DGL, t(logits), eids=ti(eids))
    assert_close(ours, theirs, atol=1e-12)


@pytest.mark.parametrize("heads", [1, 3, 8])
@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax_multi_head_matches_upstream(heads, norm_by):
    """Upstream's shape is (E, *, 1): each trailing column is its own softmax."""
    logits = rng.normal(size=(len(U), heads, 1))
    ours = ops.edge_softmax(G_MOJO, logits, norm_by=norm_by)
    theirs = dgl_ops.edge_softmax(G_DGL, t(logits), norm_by=norm_by)
    assert ours.shape == logits.shape
    assert_close(ours, theirs, atol=1e-12, label="heads={}".format(heads))


def test_edge_softmax_multi_head_on_a_subset_of_edges():
    logits = rng.normal(size=(len(U), 4, 1))
    eids = np.array([0, 5, 17, 30, 100, 511], dtype=np.int64)
    assert_close(
        ops.edge_softmax(G_MOJO, logits, eids=eids),
        dgl_ops.edge_softmax(G_DGL, t(logits), eids=ti(eids)),
        atol=1e-12,
    )


def test_edge_softmax_rejects_a_shape_without_the_trailing_axis():
    with pytest.raises(dgl_mojo.DGLError):
        ops.edge_softmax(G_MOJO, np.ones((len(U), 3)))



def test_ops_on_a_graph_with_no_edges():
    """A zero-edge graph is a shape, not a crash, on both sides."""
    empty = np.array([], dtype=np.int64)
    ours = dgl_mojo.graph((empty, empty))
    theirs = dgl.graph((ti(empty), ti(empty)))
    assert ours.num_nodes() == theirs.num_nodes() == 0
    assert ops.u_mul_e_sum(ours, np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 4)
    assert ops.edge_softmax(ours, np.zeros((0, 1))).shape == (0, 1)
    assert_close(
        ops.u_mul_e_sum(ours, np.zeros((0, 4)), np.zeros((0, 4))),
        dgl_ops.u_mul_e_sum(theirs, t(np.zeros((0, 4))), t(np.zeros((0, 4)))),
    )


def test_gspmm_on_nodes_with_no_edges_at_all():
    """Every destination is zero in-degree, so every row is the reduce identity."""
    u = np.array([0, 1], dtype=np.int64)
    v = np.array([1, 2], dtype=np.int64)
    x = rng.normal(size=(5, 4))
    e = rng.normal(size=(2, 4))
    ours = dgl_mojo.graph((u, v), num_nodes=5)
    theirs = dgl.graph((ti(u), ti(v)), num_nodes=5)
    for name, args in (
        ("u_add_e_sum", (x, e)),
        ("u_mul_e_mean", (x, e)),
        ("copy_u_max", (x,)),
        ("copy_e_min", (e,)),
    ):
        assert_close(
            getattr(ops, name)(ours, *args),
            getattr(dgl_ops, name)(theirs, *(t(a) for a in args)),
            atol=1e-12,
            label=name,
        )


def test_ops_reject_features_with_the_wrong_number_of_rows():
    """Upstream's check.h refuses these; the kernels index with no bounds test."""
    short_node = np.ones((N - 1, 4))
    with pytest.raises(Exception):
        ops.u_add_e_sum(G_MOJO, short_node, E)
    with pytest.raises(Exception):
        ops.copy_u_max(G_MOJO, short_node)
    with pytest.raises(Exception):
        ops.copy_e_sum(G_MOJO, E[:-1])
    with pytest.raises(Exception):
        ops.u_add_v(G_MOJO, short_node, X)
    with pytest.raises(Exception):
        ops.v_add_e(G_MOJO, X, E[:-1])

def test_edge_softmax_rejects_a_bad_norm_by():
    with pytest.raises(dgl_mojo.DGLError):
        ops.edge_softmax(G_MOJO, np.ones(len(U)), norm_by="edge")


def test_edge_softmax_sums_to_one_per_node():
    logits = rng.normal(size=(len(U), 1))
    score = ops.edge_softmax(G_MOJO, logits)
    deg = G_MOJO.in_degrees()
    sums = np.zeros(N)
    np.add.at(sums, G_MOJO.edges().v, score.ravel())
    np.testing.assert_allclose(sums[deg > 0], 1.0, atol=1e-12)


# The kernels take the vector path when the row width is a multiple of the SIMD
# width and the scalar tail otherwise, so every width below has to agree with
# upstream either way. 3, 5, 7 and 33 all leave a tail.
@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 8, 16, 33])
def test_gspmm_matches_upstream_at_every_feature_width(d):
    x = rng.normal(size=(N, d))
    e = rng.normal(size=(len(U), d))
    for name, args in (
        ("u_add_e_sum", (x, e)),
        ("u_div_e_min", (x, e)),
        ("u_mul_e_mean", (x, e)),
        ("copy_u_max", (x,)),
        ("copy_e_sum", (e,)),
    ):
        ours = getattr(ops, name)(G_MOJO, *args)
        theirs = getattr(dgl_ops, name)(G_DGL, *(t(a) for a in args))
        assert_close(ours, theirs, atol=1e-11, label="{} d={}".format(name, d))


@pytest.mark.parametrize("d", [1, 3, 4, 33])
def test_gsddmm_matches_upstream_at_every_feature_width(d):
    x = rng.normal(size=(N, d))
    e = rng.normal(size=(len(U), d))
    for name, args in (
        ("u_mul_v", (x, x)),
        ("u_add_e", (x, e)),
        ("u_dot_e", (x, e)),
    ):
        ours = getattr(ops, name)(G_MOJO, *args)
        theirs = getattr(dgl_ops, name)(G_DGL, *(t(a) for a in args))
        assert_close(ours, theirs, atol=1e-10, label="{} d={}".format(name, d))


@pytest.mark.parametrize("d", [1, 3, 5, 8, 33])
@pytest.mark.parametrize("reducer", ["sum", "max", "min", "mean"])
def test_segment_reduce_matches_upstream_at_every_width(d, reducer):
    seglen = np.array([3, 0, 5, 4, 2])
    value = rng.normal(size=(14, d))
    assert_close(
        ops.segment_reduce(seglen, value, reducer),
        dgl_ops.segment_reduce(ti(seglen), t(value), reducer),
        atol=1e-11,
        label="d={}".format(d),
    )


# The fused kernel keeps the max, the exponential sum and the divide in one
# pass over the segment; the same arithmetic as the five-pass composition the
# Python side used to run.
@pytest.mark.parametrize("d", [1, 3, 4, 32])
@pytest.mark.parametrize("seglen", [[1, 0, 5, 4], [0, 0, 14], [2] * 7, [14]])
def test_segment_softmax_matches_upstream(d, seglen):
    seglen = np.asarray(seglen, dtype=np.int64)
    value = rng.normal(size=(seglen.sum(), d))
    assert_close(
        ops.segment_softmax(seglen, value),
        dgl_ops.segment_softmax(ti(seglen), t(value)),
        atol=1e-12,
        label="d={} seglen={}".format(d, seglen.size),
    )


def test_segment_softmax_sums_to_one_per_segment():
    seglen = np.array([3, 0, 5, 4, 2])
    value = rng.normal(size=(14, 6))
    soft = ops.segment_softmax(seglen, value)
    np.testing.assert_allclose(soft[:3].sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(soft[3:8].sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("d2", [1, 3, 4, 5, 8, 9])
def test_gather_mm_matches_upstream_at_every_output_width(d2):
    """`d2` at and around the SIMD width: vector body, scalar tail, both."""
    n_rows, d1, r = 300, 5, 7
    a = rng.normal(size=(n_rows, d1))
    b = rng.normal(size=(r, d1, d2))
    idx = rng.integers(0, r, size=n_rows).astype(np.int64)
    assert_close(
        ops.gather_mm(a, b, idx_b=idx),
        dgl_ops.gather_mm(t(a), t(b), idx_b=ti(idx)),
        atol=1e-10,
        label="d2={}".format(d2),
    )


@pytest.mark.parametrize("d2", [1, 3, 4, 5, 8, 9])
def test_segment_mm_matches_upstream_at_every_output_width(d2):
    """`d2` at and around the SIMD width: vector body, scalar tail, both."""
    seglen = np.array([10, 5, 0, 3])
    a = rng.normal(size=(18, 5))
    b = rng.normal(size=(4, 5, d2))
    assert_close(
        ops.segment_mm(a, b, seglen),
        dgl_ops.segment_mm(t(a), t(b), ti(seglen)),
        atol=1e-10,
        label="d2={}".format(d2),
    )


def test_gather_mm_routes_a_big_block_through_segment_mm(monkeypatch):
    """The route that sorts by `idx_b` must agree with the fused kernel."""
    monkeypatch.setattr(gather_mm_mod, "_SEGMENT_BLOCK", 0)
    rng2 = np.random.default_rng(3)
    n, d1, d2, r = 1200, 9, 7, 6
    a = rng2.normal(size=(n, d1))
    b = rng2.normal(size=(r, d1, d2))
    idx = rng2.integers(0, r, size=n).astype(np.int64)
    assert_close(
        ops.gather_mm(a, b, idx_b=idx),
        dgl_ops.gather_mm(t(a), t(b), idx_b=ti(idx)),
        atol=1e-9,
    )


def test_gather_mm_routes_both_ways_to_the_same_numbers(monkeypatch):
    """The block-size cut is a performance choice, not a semantic one."""
    n, d1, d2, r = 900, 4, 4, 5
    a = rng.normal(size=(n, d1))
    b = rng.normal(size=(r, d1, d2))
    idx = rng.integers(0, r, size=n).astype(np.int64)
    fused = ops.gather_mm(a, b, idx_b=idx)
    monkeypatch.setattr(gather_mm_mod, "_SEGMENT_BLOCK", 0)
    assert_close(ops.gather_mm(a, b, idx_b=idx), fused, atol=1e-12)


# numpy hands out 16-byte-aligned buffers, so the vector loads these kernels
# issue are not 32-byte aligned. An offset view moves the base by one element
# and must still agree with upstream.
def test_offset_feature_views_match_upstream():
    # A contiguous view whose base sits one element into its allocation: the
    # kernels' vector loads start at an address no vector boundary.
    xbuf = rng.normal(size=N * 5 + 1)
    x = xbuf[1:].reshape(N, 5)
    ebuf = rng.normal(size=len(U) * 5 + 1)
    e = ebuf[1:].reshape(len(U), 5)
    assert x.flags["C_CONTIGUOUS"] and e.flags["C_CONTIGUOUS"]
    assert_close(
        ops.u_add_e_sum(G_MOJO, x, e),
        dgl_ops.u_add_e_sum(G_DGL, t(x), t(e)),
        atol=1e-11,
    )
    assert_close(
        ops.copy_u_max(G_MOJO, x),
        dgl_ops.copy_u_max(G_DGL, t(x)),
        atol=1e-11,
    )
    seglen = np.array([3, 0, 5, 4, 2])
    vbuf = rng.normal(size=14 * 4 + 1)
    value = vbuf[1:].reshape(14, 4)
    assert_close(
        ops.segment_reduce(seglen, value, "sum"),
        dgl_ops.segment_reduce(ti(seglen), t(value), "sum"),
        atol=1e-11,
    )
    assert_close(
        ops.segment_softmax(seglen, value),
        dgl_ops.segment_softmax(ti(seglen), t(value)),
        atol=1e-12,
    )


def test_zero_degree_nodes_are_zero_not_the_mean_divisor():
    g = dgl_mojo.graph(([0, 0], [1, 1]), num_nodes=4)
    out = ops.u_add_e_mean(g, np.ones((4, 4)), np.ones((2, 4)))
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[1], 2.0, atol=1e-12)
    np.testing.assert_allclose(out[2:], 0.0)


def test_node_ids_must_be_in_range():
    with pytest.raises(Exception):
        dgl_mojo.graph(([0, 1], [1, 5]), num_nodes=3)
    with pytest.raises(Exception):
        dgl_mojo.graph(([-1, 1], [1, 2]))


def test_gather_mm_rejects_out_of_range_index():
    a = np.ones((4, 3))
    b = np.ones((2, 3, 5))
    with pytest.raises(Exception):
        ops.gather_mm(a, b, idx_b=np.array([0, 1, 2, 5]))
    with pytest.raises(Exception):
        ops.gather_mm(a, b, idx_b=np.array([0, 1, 2, -1]))


# The kernels index their operands with no bounds test, so every shape the walk
# depends on is checked on the Python side. These are the regressions: each of
# these used to read past the end of a buffer or segfault.
def test_segment_mm_rejects_a_length_array_that_does_not_match_b():
    """One segment length per `b` block. Upstream raises IndexError here too."""
    a = np.ones((6, 3))
    b = np.ones((4, 3, 2))
    for seglen in (np.array([2, 2, 2]), np.array([1, 1, 1, 1, 1, 1])):
        with pytest.raises(dgl_mojo.DGLError):
            ops.segment_mm(a, b, seglen)


def test_segment_mm_rejects_negative_or_overrunning_lengths():
    """The kernel steps `row` by each length until it passes `n`, unbounded."""
    a = np.ones((6, 3))
    b = np.ones((4, 3, 2))
    with pytest.raises(dgl_mojo.DGLError):
        ops.segment_mm(a, b, np.array([-3, 3, 3, 3]))
    with pytest.raises(dgl_mojo.DGLError):
        ops.segment_mm(a, b, np.array([10, 0, 0, 0]))
    # exactly n in total is the boundary, and it is accepted
    np.testing.assert_allclose(
        ops.segment_mm(a, b, np.array([6, 0, 0, 0])), np.full((6, 2), 3.0)
    )


def test_gather_mm_rejects_a_b_block_narrower_than_a():
    """`b` blocks are `b_d1 * D2` doubles and are gathered whole."""
    a = np.arange(5 * 9, dtype=np.float64).reshape(5, 9)
    b = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    with pytest.raises(dgl_mojo.DGLError):
        ops.gather_mm(a, b, idx_b=np.array([0, 1, 0, 1, 0]))


def test_gather_mm_rejects_an_index_of_the_wrong_length():
    a = np.ones((4, 3))
    b = np.ones((2, 3, 5))
    with pytest.raises(dgl_mojo.DGLError):
        ops.gather_mm(a, b, idx_b=np.array([0, 1, 2]))
    with pytest.raises(dgl_mojo.DGLError):
        ops.gather_mm(a, b, idx_b=np.zeros((4, 1), dtype=np.int64))


def test_edge_softmax_rejects_out_of_range_eids():
    g = dgl_mojo.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
    logits = np.ones(3)
    with pytest.raises(Exception):
        ops.edge_softmax(g, logits, eids=np.array([0, 3]))


def test_complex_features_are_rejected_rather_than_truncated():
    """The kernels are Float64; dropping the imaginary part would be silent."""
    g = dgl_mojo.graph(([0, 1, 2, 0], [1, 2, 0, 2]))
    x = np.ones((3, 2), dtype=np.complex128)
    e = np.ones((4, 2), dtype=np.complex128)
    with pytest.raises(dgl_mojo.DGLError):
        ops.u_mul_e_sum(g, x, e)
    with pytest.raises(dgl_mojo.DGLError):
        ops.edge_softmax(g, np.ones(4, dtype=np.complex128))


def test_an_integer_wider_than_float64_is_rejected():
    """float64 holds 53 mantissa bits; a uint64 that does not round is refused."""
    a = np.full((4, 2), 2 ** 53 + 1, dtype=np.uint64)
    b = np.full((2, 2, 3), 2 ** 53 + 1, dtype=np.uint64)
    with pytest.raises(dgl_mojo.DGLError):
        ops.gather_mm(a, b, idx_b=np.zeros(4, dtype=np.int64))
    # one that does land on a float64 exactly is accepted
    half = float(2 ** 53)
    got = ops.gather_mm(
        np.full((2, 2), 2 ** 53, dtype=np.uint64),
        np.full((2, 2, 3), 2 ** 53, dtype=np.uint64),
        idx_b=np.zeros(2, dtype=np.int64),
    )
    # each output column is the contraction of d1=2 copies of half * half
    np.testing.assert_allclose(got, np.full((2, 3), 2.0 * half * half))


def test_node_ids_must_be_whole_numbers():
    """A fractional node id would be truncated to a different node."""
    with pytest.raises(dgl_mojo.DGLError):
        dgl_mojo.graph((np.array([0.0, 1.5]), np.array([1.0, 2.0])))
