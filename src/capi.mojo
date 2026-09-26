"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses a
parameterised function and a pointer with an inferred origin is parameterised.
Callers own all memory, including scratch, so nothing here allocates and nothing
here can leak. The kernels stay in `ported.mojo`; this file only exists because
of the divergence.

This is also the only place that still has the addresses, so it is where the
`vec` flag is decided. It is set when every row width a kernel steps by is a
multiple of `W`, so each vector covers whole columns of one row and the
remainder is the scalar tail. Base addresses are deliberately *not* part of
the test: numpy hands out 16-byte-aligned buffers, so a 32-byte gate would
switch the vector path off for every real caller. The loads are element
aligned and this toolchain emits plain unaligned vector moves for them, which
`tests/test_ops.py` exercises through deliberately offset views.
"""

from ported import (
    W,
    clamp_i64,
    cumsum,
    edge_softmax_,
    gather_mm,
    gsddmm,
    gspmm,
    replace_inf_with_zero,
    segment_mm,
    segment_reduce,
    segment_softmax,
)

def p(addr: Int) -> Pointer[Float64, AnyOrigin[mut=True]]:
    return Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)


def q(addr: Int) -> Pointer[Int, AnyOrigin[mut=True]]:
    return Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=addr)


def r(addr: Int) -> Pointer[Int8, AnyOrigin[mut=True]]:
    return Pointer[Int8, AnyOrigin[mut=True]](unsafe_from_address=addr)


# ------------------------------------------------------------------- array
@export("mdgl_cumsum")
def mdgl_cumsum(src: Int, dst: Int, n: Int) abi("C"):
    cumsum(q(src), q(dst), n)


@export("mdgl_replace_inf_with_zero")
def mdgl_replace_inf_with_zero(src: Int, dst: Int, n: Int) abi("C"):
    replace_inf_with_zero(p(src), p(dst), n)


@export("mdgl_clamp_i64")
def mdgl_clamp_i64(src: Int, dst: Int, n: Int, lo: Int, hi: Int) abi("C"):
    clamp_i64(q(src), q(dst), n, lo, hi)


# ------------------------------------------------------------------- sddmm
@export("mdgl_gsddmm")
def mdgl_gsddmm(
    uids: Int,
    vids: Int,
    lhs: Int,
    rhs: Int,
    dst: Int,
    num_edges: Int,
    lhs_target: Int,
    rhs_target: Int,
    op: Int,
    dl: Int,
    dr: Int,
    dout: Int,
) abi("C"):
    var vec = dl % W == 0 and dr % W == 0
    if op != 4:
        vec = vec and dout % W == 0
    gsddmm(
        q(uids), q(vids), p(lhs), p(rhs), p(dst), num_edges, lhs_target, rhs_target,
        op, dl, dr, dout, vec,
    )


# -------------------------------------------------------------------- spmm
@export("mdgl_gspmm")
def mdgl_gspmm(
    indptr: Int,
    indices: Int,
    eid: Int,
    lhs: Int,
    rhs: Int,
    dst: Int,
    num_dst: Int,
    op: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    reduce_op: Int,
    deg: Int,
) abi("C"):
    var vec = dl % W == 0 and dr % W == 0 and dout % W == 0
    gspmm(
        q(indptr), q(indices), q(eid), p(lhs), p(rhs), p(dst), num_dst, op, dl, dr,
        dout, reduce_op, q(deg), vec,
    )


# ----------------------------------------------------------------- segment
@export("mdgl_segment_reduce")
def mdgl_segment_reduce(
    offsets: Int,
    value: Int,
    dst: Int,
    num_seg: Int,
    d: Int,
    reduce_op: Int,
    n: Int,
) abi("C"):
    segment_reduce(q(offsets), p(value), p(dst), num_seg, d, reduce_op, n, d % W == 0)


@export("mdgl_segment_softmax")
def mdgl_segment_softmax(
    offsets: Int, value: Int, dst: Int, scratch: Int, num_seg: Int, d: Int, n: Int
) abi("C"):
    segment_softmax(
        q(offsets), p(value), p(dst), p(scratch), num_seg, d, n, d % W == 0
    )


@export("mdgl_segment_mm")
def mdgl_segment_mm(
    a: Int, b: Int, dst: Int, seglen_a: Int, n: Int, d1: Int, r: Int, d2: Int
) abi("C"):
    segment_mm(p(a), p(b), p(dst), q(seglen_a), n, d1, r, d2, d2 % W == 0)


# ------------------------------------------------------------- edge_softmax
@export("mdgl_edge_softmax_")
def mdgl_edge_softmax_(
    indptr: Int,
    indices: Int,
    eid: Int,
    mask: Int,
    logits: Int,
    dst: Int,
    num_nodes: Int,
    all_ones: Int,
) abi("C"):
    edge_softmax_(
        q(indptr), q(indices), q(eid), r(mask), p(logits), p(dst), num_nodes, all_ones
    )


# --------------------------------------------------------------- gather_mm
@export("mdgl_gather_mm")
def mdgl_gather_mm(a: Int, b: Int, idx_b: Int, dst: Int, n: Int, d1: Int, d2: Int) abi(
    "C"
):
    gather_mm(p(a), p(b), q(idx_b), p(dst), n, d1, d2)
