"""Segment aggregation operators implemented using DGL graph."""

import numpy as np

from .._lib import REDUCERS, addr, f64, i64, lib
from ..base import DGLError
from .sddmm import _flat

__all__ = ["segment_reduce", "segment_softmax", "segment_mm"]


def _cumsum(values):
    values = i64(values)
    dst = np.zeros(values.shape, dtype=np.int64)
    lib().mdgl_cumsum(addr(values), addr(dst), values.size)
    return dst


def _offsets(seglen):
    seglen = i64(seglen)
    return _cumsum(np.concatenate([np.zeros(1, dtype=np.int64), seglen]))


def _clamp(src, lo, hi):
    src = i64(src)
    dst = np.zeros(src.shape, dtype=np.int64)
    lib().mdgl_clamp_i64(addr(src), addr(dst), src.size, lo, hi)
    return f64(dst)


def _check_offsets(offsets, n):
    if offsets.size and (int(offsets[-1]) != n or bool(np.any(np.diff(offsets) < 0))):
        raise DGLError(
            "segment lengths must be non-negative and sum to {}, got offsets "
            "ending at {} for {} rows".format(
                n, int(offsets[-1]) if offsets.size else 0, n
            )
        )


def _core(reducer, value, offsets):
    value, d = _flat(value)
    offsets = i64(offsets)
    num_seg = offsets.size - 1
    _check_offsets(offsets, len(value))
    dst = np.empty((num_seg, d), dtype=np.float64)
    if num_seg == 0 or value.size == 0:
        return dst
    lib().mdgl_segment_reduce(
        addr(offsets), addr(value), addr(dst), num_seg, d, REDUCERS[reducer], value.shape[0]
    )
    return dst


def _replace_inf_with_zero(x):
    x = f64(x)
    dst = np.zeros(x.shape, dtype=np.float64)
    lib().mdgl_replace_inf_with_zero(addr(x), addr(dst), x.size)
    return dst


def segment_reduce(seglen, value, reducer="sum"):
    """Segment reduction operator.

    It aggregates the value tensor along the first dimension by segments.
    The first argument ``seglen`` stores the length of each segment. Its
    summation must be equal to the first dimension of the ``value`` tensor.
    Zero-length segments are allowed.

    Parameters
    ----------
    seglen : Tensor
        Segment lengths.
    value : Tensor
        Value to aggregate.
    reducer : str, optional
        Aggregation method. Can be 'sum', 'max', 'min', 'mean'.

    Returns
    -------
    Tensor
        Aggregated tensor of shape ``(len(seglen), value.shape[1:])``.

    Examples
    --------

    >>> import dgl
    >>> val = th.ones(10, 3)
    >>> seg = th.tensor([1, 0, 5, 4])  # 4 segments
    >>> dgl.segment_reduce(seg, val)
    tensor([[1., 1., 1.],
            [0., 0., 0.],
            [5., 5., 5.],
            [4., 4., 4.]])
    """
    seglen = i64(seglen)
    offsets = _offsets(seglen)
    value = f64(value)
    if reducer == "mean":
        rst = _core("sum", value, offsets)
        rst_shape = rst.shape
        z = _clamp(seglen, 1, len(value))
        z_shape = (rst_shape[0],) + (1,) * (len(rst_shape) - 1)
        rst /= z.reshape(z_shape)
        return rst
    elif reducer in ["min", "sum", "max"]:
        rst = _core(reducer, value, offsets)
        if reducer in ["min", "max"]:
            rst = _replace_inf_with_zero(rst)
        return rst
    else:
        raise DGLError("reducer {} not recognized.".format(reducer))


def segment_softmax(seglen, value):
    """Performa softmax on each segment.

    The first argument ``seglen`` stores the length of each segment. Its
    summation must be equal to the first dimension of the ``value`` tensor.
    Zero-length segments are allowed.

    Parameters
    ----------
    seglen : Tensor
        Segment lengths.
    value : Tensor
        Value to aggregate.

    Returns
    -------
    Tensor
        Result tensor of the same shape as the ``value`` tensor.
    """
    seglen = i64(seglen)
    shape = np.shape(value)
    value, d = _flat(value)
    n = len(value)
    offsets = _offsets(seglen)
    _check_offsets(offsets, n)
    num_seg = seglen.size
    # the kernel writes every row of `dst` once the offsets have been checked;
    # with no segments there is no row to write, so those rows are left at zero
    dst = np.zeros(value.shape, dtype=np.float64)
    if num_seg == 0 or n == 0:
        return dst.reshape(shape)
    # One kernel, three passes over the segment's own rows: the per-column max
    # and the running sum of exponentials live in two `d`-long scratch rows.
    # Upstream composes the same result out of five passes and two (n, d)
    # temporaries; the arithmetic and its order are unchanged.
    scratch = np.empty(2 * d, dtype=np.float64)
    lib().mdgl_segment_softmax(
        addr(offsets), addr(value), addr(dst), addr(scratch), num_seg, d, n
    )
    return dst.reshape(shape)


def segment_mm(a, b, seglen_a):
    r"""Performs matrix multiplication according to segments.

    Suppose ``seglen_a == [10, 5, 0, 3]``, the operator will perform
    four matrix multiplications::

        a[0:10] @ b[0], a[10:15] @ b[1],
        a[15:15] @ b[2], a[15:18] @ b[3]

    Parameters
    ----------
    a : Tensor
        The left operand, 2-D tensor of shape ``(N, D1)``
    b : Tensor
        The right operand, 3-D tensor of shape ``(R, D1, D2)``
    seglen_a : Tensor
        An integer tensor of shape ``(R,)``. Each element is the length of segments
        of input ``a``. The summation of all elements must be equal to ``N``.

    Returns
    -------
    Tensor
        The output dense matrix of shape ``(N, D2)``
    """
    a = f64(a)
    b = f64(b)
    if a.ndim != 2 or b.ndim != 3:
        raise DGLError("segment_mm wants a 2-D left operand and a 3-D right operand")
    n, d1 = a.shape
    r, b_d1, d2 = b.shape
    if b_d1 != d1:
        raise DGLError("inner dimensions do not match: {} vs {}".format(d1, b_d1))
    seglen_a = i64(seglen_a)
    dst = np.zeros((n, d2), dtype=np.float64)
    if n == 0:
        return dst
    lib().mdgl_segment_mm(addr(a), addr(b), addr(dst), addr(seglen_a), n, d1, r, d2)
    return dst
