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
        The input edge feature, of shape ``(E,)``, ``(E, 1)`` or ``(E, *, 1)``.
        Every trailing column is an independent softmax, as upstream.
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
    shape = logits.shape
    # upstream's shape is (E, *, 1): the edge axis, any number of trailing
    # columns, each its own softmax, and a final axis of one. A bare (E,) is the
    # same thing with no trailing columns. The column count is spelled out
    # rather than left to a -1, which a zero-edge array cannot resolve.
    if len(shape) == 1:
        d = 1
    elif len(shape) >= 2 and shape[-1] == 1:
        d = int(np.prod(shape[1:]))
    else:
        raise DGLError(
            "logits must be (E,), (E, 1) or (E, *, 1); got shape {}".format(shape)
        )
    logits = logits.reshape(shape[0], d)
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
    if order is None:
        # `eids` is ALL: the CSR is the whole graph's, and it is walked in edge
        # id order, so `eid` is the identity and `mask` is never read.
        row_ids = eid
        row_ptr = indptr
        row_off = indices
        all_ones = 1
        mask = np.ones(1, dtype=np.int8)
        num_rows = num_nodes
    else:
        # Upstream does not mask the graph's CSR. It rebuilds one from the
        # selection, and that rebuild indexes `logits` by POSITION in `eids`
        # rather than by edge id: `Edge_softmax_forward` calls `CalcBcastOff`
        # and then `Edge_softmax_csr_forward` on a CSRMatrix whose row length is
        # the number of selected edges sharing that endpoint, so row `i` of the
        # kernel sees `logits[i]` rather than `logits[eids[i]]`. The result is
        # indexed by position too.
        #
        # Reproduced exactly: group the selection's POSITIONS by endpoint, keep
        # them in the order they appear in `eids`, and read the logit of
        # position `i` from `logits[i]`. Both `norm_by` share the bug, because
        # the two directions differ only in which endpoint is the group key.
        uids, vids = graph._edge_arrays()
        key = vids if norm_by == "dst" else uids
        pos_of_node = {}
        row_order = []
        row_ptr = [0]
        for i, e in enumerate(order.tolist()):
            n = int(key[e])
            if n not in pos_of_node:
                pos_of_node[n] = []
            pos_of_node[n].append(i)
        for n in sorted(pos_of_node):
            grp = pos_of_node[n]
            row_order.extend(grp)
            row_ptr.append(row_ptr[-1] + len(grp))
        row_order = np.array(row_order, dtype=np.int64)
        row_ptr = np.array(row_ptr, dtype=np.int64)
        num_rows = len(pos_of_node)
        # `indices` are the CSR's neighbour slots and are not read on this path;
        # `eid` and `mask` both become the position itself, which is what the
        # kernel uses to find the logit row and to decide membership.
        row_off = row_order
        row_ids = row_order
        all_ones = 1
        mask = np.ones(1, dtype=np.int8)
        # upstream reads the selection, so the logit block it reduces over is
        # the first `len(eids)` rows of `logits`.
        logits = logits[: order.size]
    dst = np.empty((num_edges, d), dtype=np.float64)
    workers = lib().mdgl_workers()
    # Two `d`-long rows per worker for the kernel to reduce into: the
    # per-column max, then the running exponential sum for the node it is on.
    # One node at a time is enough per worker, and the workers split the nodes.
    scratch = np.empty(workers * 2 * d, dtype=np.float64)
    lib().mdgl_edge_softmax_(
        addr(row_ptr), addr(row_off), addr(row_ids), addr(mask), addr(logits),
        addr(dst), addr(scratch), num_rows, all_ones, d,
    )
    if order is not None:
        picked = np.zeros((out_len, d), dtype=np.float64)
        picked[: order.size] = dst[: order.size]
        dst = picked
    return dst.reshape(shape)
