"""Graph construction, mirroring `dgl/convert.py`."""

from __future__ import annotations

import numpy as np

from .base import DGLError
from .heterograph import DGLGraph

__all__ = ["graph"]


def graph(
    data,
    *,
    num_nodes=None,
    idtype=None,
    device=None,
    row_sorted=False,
    col_sorted=False,
):
    """Create a graph from an edge tuple or a sparse matrix.

    Parameters
    ----------
    data : tuple of arrays or scipy.sparse matrix
        Either a pair of ``(u, v)`` node-id arrays, or a scipy sparse matrix
        whose nonzero entries are the edges.
    num_nodes : int, optional
        The number of nodes. Inferred from the data when omitted.
    idtype : numpy dtype, optional
        Kept for signature compatibility; ids are always int64 here.
    device : optional
        Kept for signature compatibility; this port is CPU only.
    row_sorted : bool, optional
        Kept for signature compatibility; edges keep the order they were given.
    col_sorted : bool, optional
        Kept for signature compatibility; edges keep the order they were given.

    Returns
    -------
    DGLGraph

    Examples
    --------
    >>> import mojodgl as dgl
    >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    >>> g.num_nodes()
    5
    """
    if hasattr(data, "tocoo"):
        coo = data.tocoo()
        u, v = coo.row, coo.col
        if num_nodes is None:
            num_nodes = max(data.shape)
    elif isinstance(data, tuple) and len(data) == 2:
        u, v = data
        u = np.asarray(u)
        v = np.asarray(v)
        if not np.issubdtype(u.dtype, np.integer):
            raise DGLError("node ids must be integers")
        if not np.issubdtype(v.dtype, np.integer):
            raise DGLError("node ids must be integers")
    else:
        raise DGLError(
            "expect data to be a tuple of two node-id arrays or a scipy sparse matrix"
        )
    return DGLGraph(u, v, num_nodes=num_nodes, idtype=idtype or np.int64)
