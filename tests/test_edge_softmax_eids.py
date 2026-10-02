"""Regression tests for upstream's positional `eids` handling in edge_softmax.

These pin the documented divergence: with a subset `eids`, upstream's kernel is
handed a CSR built from the selection, and it indexes `logits` by POSITION in
`eids` rather than by edge id. Both `norm_by` share it.
"""
import numpy as np
import pytest

import mojodgl as dgl_mojo
import mojodgl.ops as ops
from upstream import assert_close, dgl, dgl_ops, t, ti


def build(seed, num_nodes=48, num_edges=300):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    ours = dgl_mojo.graph((u, v))
    theirs = dgl.graph((ti(u), ti(v)))
    return ours, theirs, num_edges


# The order of `eids` must not change the answer, because upstream's positions
# are in edge-id order on the ALL path and the selection is grouped by endpoint
# only through the CSR rebuild. Reversing must at least stay self-consistent.
@pytest.mark.parametrize("norm_by", ["dst", "src"])
@pytest.mark.parametrize("size", [1, 2, 3, 7, 16])
def test_edge_softmax_subset_matches_upstream_at_every_selection_size(norm_by, size):
    ours_g, theirs_g, num_edges = build(size + norm_by.__len__())
    logits = np.random.default_rng(1).normal(size=(num_edges, 1))
    eids = np.arange(size, dtype=np.int64)
    assert_close(
        ops.edge_softmax(ours_g, logits, eids=eids, norm_by=norm_by),
        dgl_ops.edge_softmax(theirs_g, t(logits), eids=ti(eids), norm_by=norm_by),
        atol=1e-12,
        label="norm_by={} size={}".format(norm_by, size),
    )


@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax_subset_reads_logits_by_position(norm_by):
    """The exact rule, stated as a test.

    Group the selection's positions by endpoint, then position i's value is the
    softmax of `logits[i]` over the positions in its group. Note this uses
    `logits[i]`, not `logits[eids[i]]`.
    """
    rng = np.random.default_rng(21)
    num_nodes, num_edges = 9, 26
    u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    ours_g = dgl_mojo.graph((u, v))
    theirs_g = dgl.graph((ti(u), ti(v)))
    logits = rng.normal(size=(num_edges, 1))
    eids = np.array([23, 4, 15, 11, 20, 3, 6], dtype=np.int64)
    key = (v if norm_by == "dst" else u).tolist()

    got = dgl_ops.edge_softmax(theirs_g, t(logits), eids=ti(eids), norm_by=norm_by)
    want = np.zeros((len(eids), 1))
    for i in range(len(eids)):
        group = [j for j in range(len(eids)) if key[int(eids[j])] == key[int(eids[i])]]
        w = logits[np.array(group)]
        e = np.exp(w - w.max())
        want[i] = (e / e.sum())[group.index(i)]
    np.testing.assert_allclose(got.numpy()[:len(eids)], want, rtol=1e-11, atol=1e-13)

    # and this port produces the same numbers
    assert_close(
        ops.edge_softmax(ours_g, logits, eids=eids, norm_by=norm_by), got, atol=1e-12
    )


@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax_subset_with_multi_head_logits(norm_by):
    rng = np.random.default_rng(5)
    num_nodes, num_edges = 7, 24
    u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    ours_g = dgl_mojo.graph((u, v))
    theirs_g = dgl.graph((ti(u), ti(v)))
    logits = rng.normal(size=(num_edges, 4, 1))
    eids = np.array([0, 7, 3, 19, 2, 11, 5], dtype=np.int64)
    assert_close(
        ops.edge_softmax(ours_g, logits, eids=eids, norm_by=norm_by),
        dgl_ops.edge_softmax(theirs_g, t(logits), eids=ti(eids), norm_by=norm_by),
        atol=1e-12,
    )


def test_edge_softmax_subset_can_repeat_an_edge_id():
    """Upstream has no dedup: the same id twice is two positions."""
    ours_g, theirs_g, num_edges = build(31)
    logits = np.random.default_rng(2).normal(size=(num_edges, 1))
    eids = np.array([4, 4, 9], dtype=np.int64)
    assert_close(
        ops.edge_softmax(ours_g, logits, eids=eids),
        dgl_ops.edge_softmax(theirs_g, t(logits), eids=ti(eids)),
        atol=1e-12,
    )


def test_edge_softmax_all_is_unaffected_by_the_position_rule():
    """With `eids` is ALL, positions and edge ids coincide."""
    ours_g, theirs_g, num_edges = build(44)
    logits = np.random.default_rng(3).normal(size=(num_edges, 1))
    assert_close(
        ops.edge_softmax(ours_g, logits),
        dgl_ops.edge_softmax(theirs_g, t(logits)),
        atol=1e-12,
    )


# The rule above, re-derived rather than restated: 200 random selections over
# random graphs, both `norm_by`, checked against upstream. The selection is
# drawn without replacement so each case has a distinct grouping, and the size
# is varied so both tiny and full-width selections are covered.
@pytest.mark.parametrize("norm_by", ["dst", "src"])
def test_edge_softmax_subset_matches_upstream_on_random_selections(norm_by):
    rng = np.random.default_rng(2024)
    checked = 0
    for case in range(100):
        num_nodes = int(rng.integers(2, 40))
        num_edges = int(rng.integers(1, 120))
        u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
        v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
        ours_g = dgl_mojo.graph((u, v))
        theirs_g = dgl.graph((ti(u), ti(v)))
        logits = rng.normal(size=(num_edges, 1))
        size = int(rng.integers(1, num_edges + 1))
        eids = rng.choice(num_edges, size=size, replace=False).astype(np.int64)
        assert_close(
            ops.edge_softmax(ours_g, logits, eids=eids, norm_by=norm_by),
            dgl_ops.edge_softmax(theirs_g, t(logits), eids=ti(eids), norm_by=norm_by),
            atol=1e-12,
            label="norm_by={} case={} n={} e={} size={}".format(
                norm_by, case, num_nodes, num_edges, size
            ),
        )
        checked += 1
    assert checked == 100