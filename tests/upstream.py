"""Shared helpers: the same array seen by mojodgl (numpy) and by dgl (torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")
dgl_ops = pytest.importorskip("dgl.ops")
dgl_fn = pytest.importorskip("dgl.function")


def t(a):
    """A numpy array as the torch tensor upstream dgl wants."""
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64))


def ti(a):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.int64))


def n(a):
    """Whatever upstream returned, back as numpy."""
    return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a)


def graph(u, v, num_nodes=None):
    """The same graph in both libraries."""
    return dgl.graph((ti(u), ti(v)), num_nodes=num_nodes), (u, v)


def random_graph(num_nodes=64, num_edges=512, seed=0, avg_deg=8):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    v = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    return u, v, num_nodes


def assert_close(ours, theirs, atol=1e-12, label=""):
    ours = np.asarray(ours)
    theirs = n(theirs)
    assert ours.shape == theirs.shape, "{}: shape {} vs {}".format(
        label, ours.shape, theirs.shape
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=atol, err_msg=label)
