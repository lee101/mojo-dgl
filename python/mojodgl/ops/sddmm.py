"""dgl sddmm operator module."""

import sys
from itertools import product

import numpy as np

from .._lib import (
    BINARY_OPS,
    OP_COPY_LHS,
    OP_COPY_RHS,

    TARGETS,
    addr,
    f64,

    lib,
)
from ..base import DGLError

__all__ = ["gsddmm", "copy_u", "copy_v", "copy_e"]


def reshape_lhs_rhs(lhs_data, rhs_data):
    r"""Expand dims so that there will be no broadcasting issues with different
    number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
    that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
    (E, 5, 3, 4)

    Parameters
    ----------
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    """
    lhs_shape = np.shape(lhs_data)
    rhs_shape = np.shape(rhs_data)
    if len(lhs_shape) != len(rhs_shape):
        max_ndims = max(len(lhs_shape), len(rhs_shape))
        lhs_pad_ndims = max_ndims - len(lhs_shape)
        rhs_pad_ndims = max_ndims - len(rhs_shape)
        new_lhs_shape = (lhs_shape[0],) + (1,) * lhs_pad_ndims + lhs_shape[1:]
        new_rhs_shape = (rhs_shape[0],) + (1,) * rhs_pad_ndims + rhs_shape[1:]
        lhs_data = np.reshape(lhs_data, new_lhs_shape)
        rhs_data = np.reshape(rhs_data, new_rhs_shape)
    return lhs_data, rhs_data


def _flat(a):
    """(n, ...) -> (n, d) contiguous, with a 1-D operand read as (n, 1)."""
    a = f64(a)
    if a.ndim == 1:
        return a.reshape(-1, 1), 1
    if a.ndim != 2:
        raise DGLError(
            "this port handles 1-D and 2-D features only, got shape {}".format(a.shape)
        )
    return a, a.shape[1]


def _broadcast(a, d, n_expected=None):
    """Widen a (n, 1) operand to (n, d) so NumPy semantics still hold."""
    if a.shape[1] == d:
        return a
    if a.shape[1] != 1:
        raise DGLError("operands are not broadcastable: {} and {}".format(a.shape, d))
    return np.repeat(a, d, axis=1)


def gsddmm(g, op, lhs_data, rhs_data, lhs_target="u", rhs_target="v"):
    r"""Generalized Sampled-Dense-Dense Matrix Multiplication interface.
    It computes edge features by :attr:`op` lhs features and rhs features.

    .. math::

        x_{e} = \phi(x_{lhs}, x_{rhs}), \forall (u,e,v)\in \mathcal{G}

    where :math:`x_{e}` is the returned feature on edges and :math:`x_u`,
    :math:`x_v` refers to :attr:`u`, :attr:`v` respectively. :math:`\phi`
    is the binary operator :attr:`op`, and :math:`\mathcal{G}` is the graph
    we apply gsddmm on: :attr:`g`. :math:`lhs` and :math:`rhs` are one of
    :math:`u,v,e`'s.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``,
        ``copy_lhs``, ``copy_rhs``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    lhs_target: str
        Choice of ``u``(source), ``e``(edge) or ``v``(destination) for left operand.
    rhs_target: str
        Choice of ``u``(source), ``e``(edge) or ``v``(destination) for right operand.

    Returns
    -------
    tensor
        The result tensor.
    """
    uids, vids = g._edge_arrays()
    num_edges = len(uids)
    if lhs_target not in TARGETS or rhs_target not in TARGETS:
        raise DGLError("lhs_target/rhs_target must be one of 'u', 'v', 'e'")

    if op == "copy_lhs":
        lhs, dl = _flat(lhs_data)
        rhs, dr = lhs, dl
        dout = dl
    elif op == "copy_rhs":
        rhs, dr = _flat(rhs_data)
        lhs, dl = rhs, dr
        dout = dr
    else:
        if op != "dot":
            lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)
        lhs, dl = _flat(lhs_data)
        rhs, dr = _flat(rhs_data)
        if op == "dot":
            if dl != dr:
                raise DGLError("dot needs matching feature widths, {} vs {}".format(dl, dr))
            dout = 1
        else:
            d = max(dl, dr)
            lhs = _broadcast(lhs, d)
            rhs = _broadcast(rhs, d)
            dl = dr = dout = d

    # The pointers are non-nullable, so a missing operand borrows the other's
    # buffer; its contents are never read on the copy paths.
    lhs_addr = addr(lhs) if lhs.size else addr(rhs)
    rhs_addr = addr(rhs) if rhs.size else addr(lhs)
    dst = np.empty((num_edges, dout), dtype=np.float64)
    if op == "copy_lhs":
        code = OP_COPY_LHS
    elif op == "copy_rhs":
        code = OP_COPY_RHS
    else:
        code = BINARY_OPS[op]
    lib().mdgl_gsddmm(
        addr(uids), addr(vids), lhs_addr, rhs_addr, addr(dst), num_edges,
        TARGETS[lhs_target], TARGETS[rhs_target], code, dl, dr, dout,
    )
    return dst


def _gen_sddmm_func(lhs_target, rhs_target, binary_op):
    name = "{}_{}_{}".format(lhs_target, binary_op, rhs_target)
    target_dict = {"u": "source node", "e": "edge", "v": "destination node"}
    lhs_str = target_dict[lhs_target]
    rhs_str = target_dict[rhs_target]
    docstring = r"""Generalized SDDMM function.
    It computes edge features by {op} {lhs} features and {rhs} features.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The {lhs} features.
    y : tensor
        The {rhs} features.

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
        op=binary_op, lhs=lhs_str, rhs=rhs_str
    )

    def func(g, x, y):
        return gsddmm(
            g, binary_op, x, y, lhs_target=lhs_target, rhs_target=rhs_target
        )

    func.__name__ = name
    func.__doc__ = docstring
    return func


def _register_sddmm_func():
    """Register sddmm functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_sddmm_func(lhs, rhs, binary_op)
                setattr(sys.modules[__name__], func.__name__, func)
                __all__.append(func.__name__)


def copy_u(g, x):
    r"""Generalized SDDMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.
    """
    return gsddmm(g, "copy_lhs", x, None)


def copy_v(g, x):
    r"""Generalized SDDMM function that copies destination node features to edges.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x : tensor
        The destination node features.

    Returns
    -------
    tensor
        The result tensor.
    """
    return gsddmm(g, "copy_rhs", None, x)


def copy_e(g, x):
    r"""Generalized SDDMM function that copies destination node features to edges."""
    return x


_register_sddmm_func()
