"""dgl edge_softmax operator module."""

import numpy as np

from .._lib import addr, f64, i64, lib
from ..base import ALL, DGLError, is_all

__all__ = ["edge_softmax"]


def edge_softmax(graph, logits, eids=ALL, norm_by="dst"):
    r"""Compute softmax over weights of incoming edges for every node.

    For a node :math:`i`, edge softmax is an operation that computes

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    By default edge softmax is normalized by destination nodes(i.e. :math:`ij`
    are incoming edges of `i` in the formula above). We also support edge
    softmax normalized by source nodes(i.e. :math:`ij` are outgoing edges of
    :math:`i` in the formula above). The former case corresponds to softmax in
    GAT and Transformer, and the latter case corresponds to softmax in Capsule
    network.

    Parameters
    ----------
    graph : DGLGraph
        The graph over which edge softmax will be performed.
    logits : numpy.ndarray
        The input edge feature, of shape ``(E,)`` or ``(E, 1)``.
    eids : numpy.ndarray or ALL, optional
        The IDs of the edges to apply edge softmax. If ALL, it will apply edge
        softmax to all edges in the graph. Default: ALL.
    norm_by : str, could be `src` or `dst`
        Normalized by source nodes or destination nodes. Default: `dst`.

    Returns
    -------
    numpy.ndarray
        Softmax value, of the same shape as ``logits``.

    Examples
    -----------
    >>> import dgl
    >>> import mojodgl
    >>> g = dgl.graph((th.tensor([0, 0, 0, 1, 1, 2]), th.tensor([0, 1, 2, 1, 2, 2])))
    >>> edge_softmax(g, np.ones((6, 1)))
    array([[1.],
           [0.5],
           [0.33333333],
           [0.5],
           [0.33333333],
           [0.33333333]])
    """
    num_edges = graph.number_of_edges()
    logits = f64(logits)
    keep_2d = logits.ndim == 2
    if keep_2d:
        if logits.shape[1] != 1:
            raise DGLError(
                "this port handles 1-D or (E, 1) edge logits only, got shape {}".format(
                    logits.shape
                )
            )
        logits = logits.reshape(-1)
    elif logits.ndim != 1:
        raise DGLError(
            "this port handles 1-D or (E, 1) edge logits only, got shape {}".format(
                logits.shape
            )
        )
    if norm_by == "dst":
        indptr, indices, eid = graph._in_csr()
        num_nodes = graph.number_of_nodes()
    elif norm_by == "src":
        indptr, indices, eid = graph._out_csr()
        num_nodes = graph.number_of_nodes()
    else:
        raise DGLError("norm_by must be 'src' or 'dst'")

    if is_all(eids):
        order = None
    else:
        order = i64(eids)
        if order.size and (int(order.min()) < 0 or int(order.max()) >= num_edges):
            # numpy would wrap a negative id and mask the wrong edge
            raise DGLError(
                "edge ids must be in [0, {}), got [{}, {})".format(
                    num_edges, int(order.min()), int(order.max()) + 1
                )
            )
    # The result is indexed by position in `eids`, but the buffer is as long as
    # `logits`, which is what upstream's C++ kernel produces.
    out_len = logits.shape[0]
    if order is None and out_len != num_edges:
        raise DGLError(
            "logits has {} rows but the graph has {} edges".format(out_len, num_edges)
        )
    if order is not None and order.size > out_len:
        raise DGLError(
            "{} edge ids selected but logits has only {} rows".format(
                order.size, out_len
            )
        )
    mask = np.ones(num_edges, dtype=np.int8)
    if order is not None:
        mask[:] = 0
        mask[order] = 1
    dst = np.empty(num_edges, dtype=np.float64)
    lib().mdgl_edge_softmax_(
        addr(indptr), addr(indices), addr(eid), addr(mask), addr(logits), addr(dst),
        num_nodes, 1 if order is None else 0,
    )
    if order is not None:
        picked = np.zeros(out_len, dtype=np.float64)
        picked[: order.size] = dst[order]
        dst = picked
    return dst.reshape(-1, 1) if keep_2d else dst
