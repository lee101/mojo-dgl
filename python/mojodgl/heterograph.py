"""Homogeneous graph and the message-passing driver, mirroring `dgl/heterograph.py`."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from ._lib import f64, i64
from .base import DGLError, is_all
from .function.base import TargetCode
from .function.message import BinaryMessageFunction, CopyMessageFunction, MessageFunction
from .function.reducer import ReduceFunction
from .ops.sddmm import gsddmm
from .ops.spmm import gspmm
from .udf import EdgeBatch, FeatureDict, NodeBatch

__all__ = ["DGLGraph"]

_EdgeTuple = namedtuple("EdgeTuple", "u v eid")
_Code2Str = TargetCode.CODE2STR


class DGLGraph(object):
    """A homogeneous graph.

    Edges are stored exactly in the order they were supplied, which is what
    upstream does, and message passing runs over the two CSR views derived from
    them.
    """

    def __init__(self, u, v, num_nodes=None, idtype=np.int64):
        u = i64(u)
        v = i64(v)
        if u.shape != v.shape:
            raise DGLError("u and v must have the same length")
        if num_nodes is None:
            num_nodes = int(max(u.max(initial=0), v.max(initial=0))) + 1
        num_nodes = int(num_nodes)
        if num_nodes < 0:
            raise DGLError("num_nodes must be non-negative, got {}".format(num_nodes))
        # every endpoint must be a node: the CSR views and the kernels index
        # node features by id with no bounds check
        if u.size and (
            int(u.max()) >= num_nodes or int(v.max()) >= num_nodes
            or int(u.min()) < 0 or int(v.min()) < 0
        ):
            raise DGLError(
                "node ids must be in [0, {}), got [{}, {}]".format(
                    num_nodes,
                    int(min(u.min(), v.min())),
                    int(max(u.max(), v.max())),
                )
            )
        self._u = u
        self._v = v
        self._num_nodes = int(num_nodes)
        self.idtype = idtype
        self._nstore = {}
        self._estore = {}
        self.ndata = FeatureDict(self._nstore, self._num_nodes, "ndata")
        self.edata = FeatureDict(self._estore, len(u), "edata")
        self._build_csr()

    # ------------------------------------------------------------------ csr
    def _build_csr(self) -> None:
        num_edges = self._u.size
        indeg = np.bincount(self._v, minlength=self._num_nodes)
        outdeg = np.bincount(self._u, minlength=self._num_nodes)
        self._indeg = indeg
        self._outdeg = outdeg
        self._indptr = np.zeros(self._num_nodes + 1, dtype=np.int64)
        np.cumsum(indeg, out=self._indptr[1:])
        self._outptr = np.zeros(self._num_nodes + 1, dtype=np.int64)
        np.cumsum(outdeg, out=self._outptr[1:])
        self._in_order = np.argsort(self._v, kind="stable")
        self._out_order = np.argsort(self._u, kind="stable")
        # The gathered endpoint arrays are the two CSR views the kernels read;
        # they are built once here instead of on every message-passing call.
        self._in_indices = i64(self._u[self._in_order])
        self._out_indices = i64(self._v[self._out_order])
        self._indptr = i64(self._indptr)
        self._outptr = i64(self._outptr)

    def _edge_arrays(self):
        return self._u, self._v

    def _in_csr(self):
        return self._indptr, self._in_indices, self._in_order

    def _out_csr(self):
        return self._outptr, self._out_indices, self._out_order

    # -------------------------------------------------------------- queries
    def number_of_nodes(self, ntype=None):
        return self._num_nodes

    def number_of_src_nodes(self, ntype=None):
        return self._num_nodes

    def number_of_dst_nodes(self, ntype=None):
        return self._num_nodes


    @property
    def canonical_etypes(self):
        return (("_N", "_E", "_N"),)

    def number_of_edges(self, etype=None):
        return int(self._u.size)

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return int(self._u.size)

    def num_src_nodes(self):
        return self._num_nodes

    def num_dst_nodes(self):
        return self._num_nodes

    def __len__(self):
        return self._num_nodes

    def __repr__(self):
        return "Graph(num_nodes={}, num_edges={},)".format(
            self._num_nodes, self.num_edges()
        )

    def __str__(self):
        return self.__repr__()

    def edges(self, form="uv", order="eid", etype=None):
        """All edges as a named tuple of (u, v, eid), in edge-id order."""
        if form not in ("uv", "all"):
            raise DGLError("form must be 'uv' or 'all'")
        if order != "eid":
            raise DGLError("this port only supports order='eid'")
        eid = np.arange(self._u.size, dtype=np.int64)
        return _EdgeTuple(self._u, self._v, eid)

    def _select(self, ids, order, on_src):
        if is_all(ids):
            sel = np.arange(self._u.size)
        else:
            key = self._u if on_src else self._v
            sel = np.nonzero(np.isin(key, i64(ids)))[0]
        if order == "srcid":
            sel = sel[np.argsort(self._u[sel], kind="stable")]
        elif order == "dstid":
            sel = sel[np.argsort(self._v[sel], kind="stable")]
        sel = sel.astype(np.int64)
        return _EdgeTuple(self._u[sel], self._v[sel], sel)

    def in_edges(self, v, form="uv", order="eid", etype=None):
        """All in-edges of `v`, or of every node when `v` is ALL."""
        return self._select(v, order, False)

    def out_edges(self, u, form="uv", order="eid", etype=None):
        """All out-edges of `u`, or of every node when `u` is ALL."""
        return self._select(u, order, True)

    def in_degrees(self, v=None):
        if v is None:
            return self._indeg
        return self._indeg[i64(v)]

    def out_degrees(self, u=None):
        if u is None:
            return self._outdeg
        return self._outdeg[i64(u)]


    # ------------------------------------------------------- message passing
    def _builtin_message(self, message_func):
        """Run a builtin message function, returning the per-edge messages."""
        if isinstance(message_func, CopyMessageFunction):
            target = _Code2Str[message_func.target]
            if target == "e":
                return f64(self.edata[message_func.in_field])
            return gsddmm(
                self,
                "copy_lhs" if target == "u" else "copy_rhs",
                self.ndata[message_func.in_field],
                None,
                lhs_target=target,
            )
        if not isinstance(message_func, BinaryMessageFunction):
            raise DGLError("unrecognised message function {}".format(message_func))
        lhs = _Code2Str[message_func.lhs]
        rhs = _Code2Str[message_func.rhs]
        lhs_data = self.edata[message_func.lhs_field] if lhs == "e" \
            else self.ndata[message_func.lhs_field]
        rhs_data = self.edata[message_func.rhs_field] if rhs == "e" \
            else self.ndata[message_func.rhs_field]
        return gsddmm(
            self, message_func.binary_op, lhs_data, rhs_data,
            lhs_target=lhs, rhs_target=rhs,
        )

    def _fusable(self, message_func):
        """The (op, lhs, rhs) triple when gspmm can do message and reduce at once."""
        if isinstance(message_func, CopyMessageFunction):
            target = _Code2Str[message_func.target]
            if target == "u":
                return "copy_lhs", "u", None, message_func.in_field, None
            if target == "e":
                return "copy_rhs", "e", None, None, message_func.in_field
            return None
        if not isinstance(message_func, BinaryMessageFunction):
            return None
        if message_func.binary_op == "dot":
            return None
        if _Code2Str[message_func.lhs] != "u" or _Code2Str[message_func.rhs] != "e":
            return None
        return (
            message_func.binary_op,
            "u",
            "e",
            message_func.lhs_field,
            message_func.rhs_field,
        )

    def _reduce_messages(self, message, reduce_op):
        return gspmm(self, "copy_rhs", reduce_op, None, message)

    def _pass(self, message_func, reduce_func, apply_node_func, write=True):
        """Run one message-passing round; returns the new node features.

        `write=False` leaves the feature store untouched, which is what `pull`
        needs: only the rows it asked for may be written back.
        """
        num_nodes = self._num_nodes
        fused = self._fusable(message_func) if isinstance(message_func, MessageFunction) \
            else None
        if fused is not None and isinstance(reduce_func, ReduceFunction):
            op, _, _, lhs_field, rhs_field = fused
            lhs_data = self.ndata[lhs_field] if lhs_field is not None else None
            rhs_data = self.edata[rhs_field] if rhs_field is not None else None
            result = {reduce_func.out_field: gspmm(self, op, reduce_func.name, lhs_data, rhs_data)}
        else:
            if isinstance(message_func, MessageFunction):
                messages = {message_func.out_field: self._builtin_message(message_func)}
            else:
                edges = EdgeBatch(self, np.arange(self._u.size, dtype=np.int64))
                messages = dict(message_func(edges))
            if isinstance(reduce_func, ReduceFunction):
                key = next(iter(messages))
                result = {
                    reduce_func.out_field: self._reduce_messages(
                        messages[key], reduce_func.name
                    )
                }
            else:
                nodes = NodeBatch(
                    data=self._nstore,
                    mailbox={
                        k: self._reduce_messages(v, "sum") for k, v in messages.items()
                    },
                    num_nodes=num_nodes,
                )
                result = dict(reduce_func(nodes))
        if apply_node_func is not None:
            # upstream hands apply_node_func the aggregated features as node
            # data; its mailbox is None.
            nodes = NodeBatch(data=result, mailbox=None, num_nodes=num_nodes)
            result = dict(apply_node_func(nodes))
        result = {k: f64(v) for k, v in result.items()}
        if write:
            for key, value in result.items():
                self.ndata[key] = value
        return result

    def update_all(self, message_func, reduce_func, apply_node_func=None, etype=None):
        """Send messages along all the edges of the specified type
        and update all the nodes of the corresponding destination type.

        Parameters
        ----------
        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction.
        etype : str or (str, str, str), optional
            The type name of the edges.

        Notes
        -----
        * If some of the nodes in the graph has no in-edges, DGL does not invoke
        message and reduce functions for these nodes and fill their aggregated messages
        with zero. DGL still invokes :attr:`apply_node_func` if provided.
        """
        if etype is not None and etype not in self.canonical_etypes:
            raise DGLError("etype {} not in {}".format(etype, self.canonical_etypes))
        self._pass(message_func, reduce_func, apply_node_func)

    def pull(self, v, message_func, reduce_func, apply_node_func=None, etype=None):
        """Pull messages from the in-edges of the given nodes and update
        the node features.

        Parameters
        ----------
        v : array-like
            The destination nodes.
        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction.
        """
        if etype is not None and etype not in self.canonical_etypes:
            raise DGLError("etype {} not in {}".format(etype, self.canonical_etypes))
        v = np.unique(i64(v))
        if v.size == 0:
            return
        mask = np.isin(self._v, v)
        sub = DGLGraph(self._u[mask], self._v[mask], num_nodes=self._num_nodes)
        for key, value in self._nstore.items():
            sub.ndata[key] = value
        for key, value in self._estore.items():
            sub.edata[key] = value[mask]
        result = sub._pass(message_func, reduce_func, apply_node_func, write=False)
        for key, value in result.items():
            rows = np.zeros_like(value)
            rows[v] = value[v]
            self.ndata[key] = rows
