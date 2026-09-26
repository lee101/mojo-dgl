"""dgl gather_mm operator module."""

import numpy as np

from .._lib import addr, f64, i64, lib
from ..base import DGLError
from .segment import segment_mm

__all__ = ["gather_mm"]


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
    R, _, D2 = b.shape
    idx_b = i64(idx_b)
    if idx_b.size and (int(idx_b.min()) < 0 or int(idx_b.max()) >= R):
        # the kernel indexes `b` with no bounds check
        raise DGLError(
            "idx_b must be in [0, {}), got [{}, {})".format(
                R, int(idx_b.min()), int(idx_b.max()) + 1
            )
        )
    if N > 1000000 or D1 > 8 or D2 > 8:
        # Use segment_mm for large workload
        sorted_idx_b = np.sort(idx_b)
        perm = np.argsort(idx_b, kind="stable")
        rev_perm = np.argsort(perm, kind="stable")
        sorted_a = a[perm]
        pos_l = np.searchsorted(sorted_idx_b, np.arange(R))
        pos_r = np.append(pos_l[1:], len(idx_b))
        seglen = pos_r - pos_l
        return segment_mm(sorted_a, b, seglen)[rev_perm]
    dst = np.zeros((N, D2), dtype=np.float64)
    if N == 0:
        return dst
    lib().mdgl_gather_mm(addr(a), addr(b), addr(idx_b), addr(dst), N, D1, D2)
    return dst
