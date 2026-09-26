"""User-defined function support, mirroring `dgl/udf.py`."""

from __future__ import annotations

import numpy as np

__all__ = ["FeatureDict", "NodeBatch", "EdgeBatch"]


class FeatureDict:
    """The `g.ndata` / `g.edata` mapping: feature name -> contiguous array."""

    def __init__(self, store: dict, num_rows: int, what: str):
        self._store = store
        self._num_rows = num_rows
        self._what = what

    def __setitem__(self, key, value):
        arr = np.ascontiguousarray(value, dtype=np.float64)
        if arr.ndim == 0 or len(arr) != self._num_rows:
            raise ValueError(
                "{} expects {} rows, got shape {}".format(
                    self._what, self._num_rows, arr.shape
                )
            )
        self._store[key] = arr

    def __getitem__(self, key):
        try:
            return self._store[key]
        except KeyError:
            raise KeyError("cannot find {} feature {}".format(self._what[:-1], key))

    def __contains__(self, key):
        return key in self._store

    def __delitem__(self, key):
        del self._store[key]

    def __len__(self):
        return len(self._store)

    def keys(self):
        return self._store.keys()

    def pop(self, key, *default):
        return self._store.pop(key, *default)

    def __repr__(self):
        return repr(self._store)


class _NodeDataView:
    """`edges.src` / `edges.dst`: node features indexed by the edge endpoints."""

    def __init__(self, graph, ids: np.ndarray):
        self._graph = graph
        self._ids = ids

    def __getitem__(self, key):
        return self._graph.ndata[key][self._ids]

    def __contains__(self, key):
        return key in self._graph.ndata

    def __repr__(self):
        return "src" if self._graph._edge_view_is_src else "dst"


class NodeBatch:
    """The `nodes` argument a reduce or apply UDF receives."""

    def __init__(self, data=None, mailbox=None, num_nodes: int = 0, batch_num_nodes=None):
        self._data = data if data is not None else {}
        self._mailbox = mailbox if mailbox is not None else {}
        self._num_nodes = num_nodes

    def __getitem__(self, key):
        if key in self._mailbox:
            return self._mailbox[key]
        return self._data[key]

    @property
    def mailbox(self):
        return self._mailbox

    @property
    def data(self):
        return self._data

    def batch_num_nodes(self, ntype=None):
        return [self._num_nodes]

    def num_nodes(self, ntype=None):
        return self._num_nodes

    def __repr__(self):
        return "NodeBatch(num_nodes={}, keys={})".format(
            self._num_nodes, list(self._mailbox) + list(self._data)
        )


class EdgeBatch:
    """The `edges` argument a message UDF receives."""

    _edge_view_is_src = True

    def __init__(self, graph, eids: np.ndarray):
        self._graph = graph
        self._eids = eids
        self._src = _NodeDataView(graph, graph._u[eids])
        self._edge_view_is_src = False
        self._dst = _NodeDataView(graph, graph._v[eids])

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def data(self):
        return _EdgeDataView(self._graph, self._eids)

    @property
    def number_of_nodes(self):
        return self._graph.number_of_nodes()

    @property
    def number_of_edges(self):
        return len(self._eids)

    def batch_num_nodes(self, ntype=None):
        return [self._graph.number_of_nodes()]

    def batch_num_edges(self, etype=None):
        return [len(self._eids)]

    def __repr__(self):
        return "EdgeBatch(num_edges={})".format(len(self._eids))


class _EdgeDataView:
    def __init__(self, graph, eids: np.ndarray):
        self._graph = graph
        self._eids = eids

    def __getitem__(self, key):
        return self._graph.edata[key][self._eids]

    def __contains__(self, key):
        return key in self._graph.edata
