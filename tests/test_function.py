"""`dgl.function` builtins and `DGLGraph.update_all`, against upstream dgl."""

import itertools

import numpy as np
import pytest

import mojodgl as dgl_mojo
import mojodgl.function as fn
import mojodgl.ops as ops
from upstream import assert_close, dgl, dgl_fn, random_graph, t, ti

U, V, N = random_graph(num_nodes=48, num_edges=400, seed=1)
rng = np.random.default_rng(11)
X = rng.normal(size=(N, 6))
E = rng.normal(size=(len(U), 6))


def pair():
    g = dgl_mojo.graph((U, V))
    g.ndata["x"] = X
    g.edata["e"] = E
    d = dgl.graph((ti(U), ti(V)))
    d.ndata["x"] = t(X)
    d.edata["e"] = t(E)
    return g, d


def field_for(target):
    return "e" if target == "e" else "x"


# ------------------------------------------------------------------ objects
def test_builtin_objects_have_upstream_names():
    for name in ("u_add_v", "u_mul_e", "v_div_u", "e_dot_v", "u_sub_e"):
        builtin = getattr(fn, name)
        assert callable(builtin)
        msg = builtin("a", "b", "m")
        assert msg.name == name
        assert msg.lhs_field == "a" and msg.rhs_field == "b" and msg.out_field == "m"
    assert fn.copy_u("a", "m").name == "copy_u"
    assert fn.copy_e("a", "m").name == "copy_e"
    for name in ("sum", "max", "min", "mean"):
        red = getattr(fn, name)("m", "h")
        assert red.name == name and red.msg_field == "m" and red.out_field == "h"


def test_all_thirty_binary_message_builtins_exist():
    expected = {
        "{}_{}_{}".format(a, op, b)
        for a, b in itertools.permutations("uve", 2)
        for op in ("add", "sub", "mul", "div", "dot")
    }
    assert expected <= set(dir(fn))
    assert expected <= set(dir(ops))
    for name in expected:
        assert getattr(ops, name).__doc__ is not None


# --------------------------------------------------------------- update_all
@pytest.mark.parametrize("name", ["copy_u", "copy_e"])
@pytest.mark.parametrize("reduce_op", ["sum", "max", "min", "mean"])
def test_update_all_copy(name, reduce_op):
    ours, theirs = pair()
    field = "x" if name == "copy_u" else "e"
    ours.update_all(getattr(fn, name)(field, "m"), getattr(fn, reduce_op)("m", "h"))
    theirs.update_all(
        getattr(dgl_fn, name)(field, "m"), getattr(dgl_fn, reduce_op)("m", "h")
    )
    assert_close(ours.ndata["h"], theirs.ndata["h"], atol=1e-11, label=name + reduce_op)


@pytest.mark.parametrize("lhs,rhs", list(itertools.permutations("uve", 2)))
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "dot"])
def test_update_all_binary_builtin(lhs, rhs, op):
    name = "{}_{}_{}".format(lhs, op, rhs)
    ours, theirs = pair()
    ours.update_all(
        getattr(fn, name)(field_for(lhs), field_for(rhs), "m"), fn.sum("m", "h")
    )
    theirs.update_all(
        getattr(dgl_fn, name)(field_for(lhs), field_for(rhs), "m"),
        dgl_fn.sum("m", "h"),
    )
    assert_close(ours.ndata["h"], theirs.ndata["h"], atol=1e-10, label=name)


def test_update_all_max_and_min_fill_zero_degree_with_zero():
    u = np.array([0, 0, 1], dtype=np.int64)
    v = np.array([1, 1, 2], dtype=np.int64)
    ours = dgl_mojo.graph((u, v))
    theirs = dgl.graph((ti(u), ti(v)))
    x = np.array([[1.0], [2.0], [3.0]])
    ours.ndata["h"] = x
    theirs.ndata["h"] = t(x)
    for reduce_op in ("max", "min", "mean", "sum"):
        ours.update_all(fn.copy_u("h", "m"), getattr(fn, reduce_op)("m", "h"))
        theirs.update_all(
            dgl_fn.copy_u("h", "m"), getattr(dgl_fn, reduce_op)("m", "h")
        )
        assert_close(ours.ndata["h"], theirs.ndata["h"], atol=1e-12)


def test_update_all_with_udf_message_function():
    ours, theirs = pair()
    ours.update_all(
        lambda edges: {"m": edges.src["x"] * edges.data["e"] + 1.0}, fn.mean("m", "h")
    )
    theirs.update_all(
        lambda edges: {"m": edges.src["x"] * edges.data["e"] + 1.0}, dgl_fn.mean("m", "h")
    )
    assert_close(ours.ndata["h"], theirs.ndata["h"], atol=1e-10)


def test_update_all_with_apply_node_func():
    ours, theirs = pair()
    ours.update_all(
        fn.copy_u("x", "m"), fn.sum("m", "h"),
        lambda nodes: {"h": nodes.data["h"] * 0.5 + 2.0},
    )
    theirs.update_all(
        dgl_fn.copy_u("x", "m"), dgl_fn.sum("m", "h"),
        lambda nodes: {"w": nodes.data["h"] * 0.5 + 2},
    )
    assert_close(ours.ndata["h"], theirs.ndata["w"], atol=1e-10)


def test_udf_reduce_receives_the_pre_summed_mailbox():
    """Documented divergence: our mailbox is (N, D) and pre-summed.

    Upstream hands a reduce UDF a ragged (N, max_in_degree, D) mailbox, so the
    two are compared against the same builtin reduction rather than to each
    other directly.
    """
    ours, theirs = pair()
    ours.update_all(
        fn.u_mul_e("x", "e", "m"), lambda nodes: {"h": nodes.mailbox["m"] * 2.0}
    )
    theirs.update_all(dgl_fn.u_mul_e("x", "e", "m"), dgl_fn.sum("m", "s"))
    assert_close(ours.ndata["h"], theirs.ndata["s"] * 2.0, atol=1e-10)


def test_pull_matches_upstream():
    targets = np.array([1, 5, 9, 40], dtype=np.int64)
    ours, theirs = pair()
    ours.pull(
        targets,
        fn.u_mul_e("x", "e", "m"),
        fn.sum("m", "h"),
        apply_node_func=lambda nodes: {"h": nodes.data["h"] + 1.0},
    )
    theirs.pull(
        ti(targets),
        dgl_fn.u_mul_e("x", "e", "m"),
        dgl_fn.sum("m", "h"),
        apply_node_func=lambda nodes: {"w": nodes.data["h"] + 1},
    )
    assert_close(ours.ndata["h"], theirs.ndata["w"], atol=1e-10)


# ------------------------------------------------- published upstream vectors
def test_update_all_docstring_example():
    g = dgl_mojo.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    g.ndata["x"] = np.ones((5, 2))
    g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
    expected = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(g.ndata["h"], expected)


def test_edge_softmax_docstring_example():
    g = dgl_mojo.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))
    edata = np.ones((6, 1))
    np.testing.assert_allclose(
        ops.edge_softmax(g, edata),
        np.array([[1.0], [0.5], [1 / 3], [0.5], [1 / 3], [1 / 3]]),
    )
    np.testing.assert_allclose(
        ops.edge_softmax(g, edata, norm_by="src"),
        np.array([[1 / 3], [1 / 3], [1 / 3], [0.5], [0.5], [1.0]]),
    )
    np.testing.assert_allclose(
        ops.edge_softmax(g, edata[:4], eids=np.array([0, 1, 2, 3])),
        np.array([[1.0], [0.5], [1.0], [0.5]]),
    )


def test_segment_docstring_example():
    val = np.ones((10, 3))
    seg = np.array([1, 0, 5, 4])
    np.testing.assert_allclose(
        ops.segment_reduce(seg, val),
        np.array([[1.0] * 3, [0.0] * 3, [5.0] * 3, [4.0] * 3]),
    )
    soft = ops.segment_softmax(seg, val)
    np.testing.assert_allclose(soft[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(soft[1], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(soft[6], [0.25, 0.25, 0.25])


def test_pull_matches_upstream():
    ours_g, theirs_g = pair()
    nodes = np.array([0, 2])
    ours_g.pull(nodes, fn.u_mul_e("x", "e", "m"), fn.sum("m", "z"))
    theirs_g.pull(nodes, dgl_fn.u_mul_e("x", "e", "m"), dgl_fn.sum("m", "z"))
    assert_close(ours_g.ndata["z"], theirs_g.ndata["z"], atol=1e-12, label="pull")


def test_nn_functional_edge_softmax_is_the_same_function():
    import mojodgl.nn.functional as nnf

    assert nnf.edge_softmax is ops.edge_softmax
