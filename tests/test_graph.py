"""DGLGraph construction and queries, checked against upstream dgl."""

import numpy as np
import pytest

import mojodgl as dgl_mojo
from upstream import dgl, ti

U = np.array([2, 0, 1, 0])
V = np.array([1, 2, 0, 1])


def test_edge_order_is_preserved():
    ours = dgl_mojo.graph((U, V))
    theirs = dgl.graph((ti(U), ti(V)))
    assert ours.num_nodes() == theirs.number_of_nodes()
    assert ours.num_edges() == theirs.number_of_edges()
    np.testing.assert_array_equal(ours.edges().u, np.asarray(theirs.edges()[0]))
    np.testing.assert_array_equal(ours.edges().v, np.asarray(theirs.edges()[1]))
    np.testing.assert_array_equal(ours.edges().eid, np.arange(len(U)))


def test_num_nodes_inference_and_override():
    ours = dgl_mojo.graph((U, V), num_nodes=10)
    theirs = dgl.graph((ti(U), ti(V)), num_nodes=10)
    assert ours.number_of_nodes() == theirs.number_of_nodes() == 10
    assert ours.number_of_src_nodes() == theirs.number_of_src_nodes()
    assert ours.number_of_dst_nodes() == theirs.number_of_dst_nodes()
    assert ours.num_src_nodes() == theirs.num_src_nodes()
    assert ours.num_dst_nodes() == theirs.num_dst_nodes()
    assert list(ours.canonical_etypes) == list(theirs.canonical_etypes)
    assert dgl_mojo.graph((U, V)).number_of_nodes() == 3


def test_degrees():
    ours = dgl_mojo.graph((U, V))
    theirs = dgl.graph((ti(U), ti(V)))
    np.testing.assert_array_equal(ours.in_degrees(), np.asarray(theirs.in_degrees()))
    np.testing.assert_array_equal(ours.out_degrees(), np.asarray(theirs.out_degrees()))
    np.testing.assert_array_equal(ours.in_degrees([0, 2]), np.asarray(theirs.in_degrees(0)))


def test_in_and_out_edges():
    ours = dgl_mojo.graph((U, V))
    theirs = dgl.graph((ti(U), ti(V)))
    for v in (0, 1, 2):
        a = ours.in_edges(v)
        b = theirs.in_edges(v, form="all")
        assert a.u.tolist() == np.asarray(b[0]).tolist()
        assert a.v.tolist() == np.asarray(b[1]).tolist()
        np.testing.assert_array_equal(a.eid, np.asarray(b[2]))
    for u in (0, 1, 2):
        a = ours.out_edges(u)
        b = theirs.out_edges(u, form="all")
        assert a.u.tolist() == np.asarray(b[0]).tolist()
        assert a.v.tolist() == np.asarray(b[1]).tolist()
        np.testing.assert_array_equal(a.eid, np.asarray(b[2]))
    a = ours.in_edges(dgl_mojo.ALL)
    assert a.u.tolist() == ours.edges().u.tolist()


def test_graph_from_scipy_sparse():
    import scipy.sparse as sp

    mat = sp.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64))
    ours = dgl_mojo.graph(mat)
    assert ours.num_nodes() == 3
    assert ours.num_edges() == 3
    assert sorted(zip(ours.edges().u.tolist(), ours.edges().v.tolist())) == [
        (0, 1), (1, 2), (2, 0)
    ]


def test_rejects_non_integer_ids():
    with pytest.raises(dgl_mojo.DGLError):
        dgl_mojo.graph((np.array([0.5]), np.array([1.0])))


def test_feature_store_rejects_wrong_row_count():
    g = dgl_mojo.graph((U, V))
    with pytest.raises(ValueError):
        g.ndata["h"] = np.ones((2, 3))
    with pytest.raises(KeyError):
        g.ndata["missing"]


def test_empty_graph_infers_zero_nodes():
    """Upstream's inferred node count for no edges is 0, not 1."""
    empty = np.array([], dtype=np.int64)
    ours = dgl_mojo.graph((empty, empty))
    theirs = dgl.graph((ti(empty), ti(empty)))
    assert ours.num_nodes() == theirs.num_nodes() == 0
    assert ours.num_edges() == theirs.num_edges() == 0
    np.testing.assert_array_equal(ours.in_degrees(), np.asarray(theirs.in_degrees()))
