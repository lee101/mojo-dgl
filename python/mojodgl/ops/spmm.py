"""Internal module for general spmm operators."""

import sys

import numpy as np

from .._lib import (
    BINARY_OPS,
    OP_COPY_LHS,
    OP_COPY_RHS,
    REDUCERS,
    addr,
    lib,
)
from ..base import DGLError
from .sddmm import _broadcast, _check_rows, _flat, reshape_lhs_rhs

__all__ = ["gspmm"]


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    r"""Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel.

    1. Computes messages by :attr:`op` source node and edge features.
    2. Aggregate the messages by :attr:`reduce_op` as the features on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e) \in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math:`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``, ``mean``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
        The result tensor.
    """
    if reduce_op not in REDUCERS:
        raise DGLError("reduce_op {} not recognized".format(reduce_op))
    if op not in list(BINARY_OPS) + ["copy_lhs", "copy_rhs"]:
        raise DGLError("op {} not recognized".format(op))
    if op == "dot":
        raise DGLError("dot is not a gspmm operator; use gsddmm")

    if op == "copy_lhs":
        lhs, dl = _flat(lhs_data)
        rhs, dr = lhs, dl
    elif op == "copy_rhs":
        rhs, dr = _flat(rhs_data)
        lhs, dl = rhs, dr
    else:
        lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)
        lhs, dl = _flat(lhs_data)
        rhs, dr = _flat(rhs_data)
        d = max(dl, dr)
        lhs = _broadcast(lhs, d)
        rhs = _broadcast(rhs, d)
        dl = dr = d
    dout = max(dl, dr)

    num_dst = g.number_of_nodes()
    indptr, indices, eid = g._in_csr()
    lhs_addr = addr(lhs) if lhs.size else addr(rhs)
    rhs_addr = addr(rhs) if rhs.size else addr(lhs)
    if op == "copy_lhs":
        code = OP_COPY_LHS
    elif op == "copy_rhs":
        code = OP_COPY_RHS
    else:
        code = BINARY_OPS[op]
    # upstream's check.h rejects an operand whose row count is not the one its
    # role implies; the kernel would read past the end of the array instead.
    # A copy op carries one array, node data for copy_lhs and edge data for
    # copy_rhs; a binary op carries both.
    num_edges = g.number_of_edges()
    if code == OP_COPY_LHS:
        _check_rows(lhs, num_dst, "node")
    elif code == OP_COPY_RHS:
        _check_rows(rhs, num_edges, "edge")
    else:
        _check_rows(lhs, num_dst, "node")
        _check_rows(rhs, num_edges, "edge")
    ret = np.empty((num_dst, dout), dtype=np.float64)
    # `mean` divides by the in-degree inside the kernel, so the reduced row is
    # written once instead of being handed back to numpy for a second pass.
    deg = g.in_degrees()
    lib().mdgl_gspmm(
        addr(indptr), addr(indices), addr(eid), lhs_addr, rhs_addr, addr(ret),
        num_dst, code, dl, dr, dout, REDUCERS[reduce_op], addr(deg),
    )
    return ret


def _attach_zerodeg_note(docstring, reducer):
    note1 = """
    The {} function will return zero for nodes with no incoming messages.""".format(
        reducer
    )
    note2 = """
    This is implemented by replacing all {} values to zero.
    """.format(
        "infinity" if reducer == "min" else "negative infinity"
    )

    docstring = docstring + note1
    if reducer in ("min", "max"):
        docstring = docstring + note2
    return docstring


def _gen_spmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = """Generalized SpMM function.
    It fuses two steps into one kernel.

    1. Computes messages by {} source node and edge features.
    2. Aggregate the messages by {} as the features on destination nodes.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The source node features.
    y : tensor
        The edge features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(
        binary_op, reduce_op
    )
    docstring = _attach_zerodeg_note(docstring, reduce_op)

    def func(g, x, y):
        return gspmm(g, binary_op, reduce_op, x, y)

    func.__name__ = name
    func.__doc__ = docstring
    return func


def _gen_copy_reduce_func(binary_op, reduce_op):
    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies node feature to edge as the message.",
        "copy_e": "It regards edge feature as message.",
    }
    x_str = {"copy_u": "source node", "copy_e": "edge"}

    def docstring(binary_op):
        return _attach_zerodeg_note(
            """Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.
    """.format(
                binary_str[binary_op], reduce_op, x_str[binary_op]
            ),
            reduce_op,
        )

    def func(g, x):
        if binary_op == "copy_u":
            return gspmm(g, "copy_lhs", reduce_op, x, None)
        return gspmm(g, "copy_rhs", reduce_op, None, x)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func


def _register_spmm_func():
    """Register spmm functions

    - Binary operation plus reduction between u and e: u_[]_e_[]
    - Copy u plus reduction: copy_u_[]
    - Copy e plus reduction: copy_e_[]
    """
    for binary_op in ["add", "sub", "mul", "div", "copy_u", "copy_e"]:
        for reduce_op in ["sum", "max", "min", "mean"]:
            if binary_op.startswith("copy"):
                func = _gen_copy_reduce_func(binary_op, reduce_op)
            else:
                func = _gen_spmm_func(binary_op, reduce_op)
            setattr(sys.modules[__name__], func.__name__, func)
            __all__.append(func.__name__)


_register_spmm_func()
