"""dgl gather_mm operator module."""

import os

import numpy as np

from .._lib import addr, f64, i64, lib
from ..base import DGLError
from .segment import segment_mm

__all__ = ["gather_mm"]


# `D1 * D2` doubles in the `b` block above which the sorted route wins. Cut
# from a sweep at 1M rows / 64 blocks: the fused kernel is ahead by 4.6x at
# 16x16, 1.6x at 32x32, 1.2x at 64x32 and 64x64, and still level at 128x128;
# the sorted route only takes it at 256x256, where the block is 32 MB.
_SEGMENT_BLOCK = int(os.environ.get("MOJODGL_GATHER_MM_SEGMENT_BLOCK", 16384))


def gather_mm(a, b, *, idx_b):
    r"""Gather data according to the given indices and perform matrix multiplication.

    Let the result tensor be ``c``, the operator conducts the following computation:

      c[i] = a[i] @ b[idx_b[i]]
      , where len(c) == len(idx_b)


    Parameters
    ----------
    a : Tensor
        A 2-D tensor of shape ``(N, D1)``
    b : Tensor
        A 3-D tensor of shape ``(R, D1, D2)``
    idx_b : Tensor, optional
        An 1-D integer tensor of shape ``(N,)``.

    Returns
    -------
    Tensor
        The output dense matrix of shape ``(N, D2)``
    """
    a = f64(a)
    b = f64(b)
    if a.ndim != 2 or b.ndim != 3:
        raise DGLError("gather_mm wants a 2-D and a 3-D operand")
    N, D1 = a.shape
    R, b_d1, D2 = b.shape
    if b_d1 != D1:
        # The kernel contracts `a`'s D1 columns against one `b` block of
        # `b_d1 * D2` doubles with no bounds test, so a narrower block would be
        # read past. Upstream's bmm rejects the same shapes.
        raise DGLError(
            "inner dimensions do not match: a is {} wide, b blocks are {}".format(
                D1, b_d1
            )
        )
    idx_b = i64(idx_b)
    if idx_b.ndim != 1 or idx_b.size != N:
        raise DGLError(
            "idx_b must be 1-D with one index per row of a: got shape {} for "
            "N={}".format(idx_b.shape, N)
        )
    if idx_b.size and (int(idx_b.min()) < 0 or int(idx_b.max()) >= R):
        # the kernel indexes `b` with no bounds check
        raise DGLError(
            "idx_b must be in [0, {}), got [{}, {})".format(
                R, int(idx_b.min()), int(idx_b.max()) + 1
            )
        )
    # Two routes compute the same product. The fused kernel streams `a` and
    # `dst` and gathers the `b` block per row; the segment route sorts the rows
    # by `idx_b` so rows sharing a block walk it once, but pays a random-access
    # gather of `a` and a random-access scatter of the result on top. The
    # fused kernel wins while the `b` block is small enough to stay resident
    # between rows, which is the attention-head shape; past that the block no
    # longer survives in cache and the sorted route's reuse wins.
    if D1 * D2 > _SEGMENT_BLOCK:
        sorted_idx_b = np.sort(idx_b)
        perm = np.argsort(idx_b, kind="stable")
        rev_perm = np.argsort(perm, kind="stable")
        sorted_a = a[perm]
        pos_l = np.searchsorted(sorted_idx_b, np.arange(R))
        pos_r = np.append(pos_l[1:], len(idx_b))
        seglen = pos_r - pos_l
        return segment_mm(sorted_a, b, seglen)[rev_perm]
    dst = np.empty((N, D2), dtype=np.float64)
    if N == 0:
        return dst
    lib().mdgl_gather_mm(addr(a), addr(b), addr(idx_b), addr(dst), N, D1, D2)
    return dst
