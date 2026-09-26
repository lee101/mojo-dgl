"""Kernels ported from DGL 2.3.0.

The sections below follow upstream's own source order: the array primitives in
`src/array/cpu/` come first because `python/dgl/ops/segment.py` is written
against them, then `python/dgl/function/`, then `python/dgl/ops/` in the order
`ops/__init__.py` imports it. Every kernel keeps upstream's argument order and
its branch structure; the only concessions to Mojo are the `Int`-addressed
buffers (see `capi.mojo`) and the fact that a `def` cannot be parameterised on
an op, so upstream's C++ functors become an integer dispatch.

Element counts are `Int`; a feature block of `n` rows and `d` columns is `n *
d` contiguous `Float64`.

The column loop of every feature kernel is vectorised at `W` lanes, with a
scalar tail for the `d % W` columns that do not fill a vector. `vec` is
decided by `capi.mojo`, which is the only place that still has the
buffer addresses: it sets it when every row width a kernel steps by is a
multiple of `W`, so each vector covers whole columns of one row and the
remainder is the scalar tail. When it is false the kernels run the scalar
path end to end.
"""

from std.math import exp
from std.sys import simd_width_of

comptime FP = Pointer[Float64, AnyOrigin[mut=True]]
comptime IP = Pointer[Int, AnyOrigin[mut=True]]
comptime BP = Pointer[Int8, AnyOrigin[mut=True]]

comptime W = simd_width_of[DType.float64]()
comptime VW = SIMD[DType.float64, W]

# `dgl/function/base.py` :: TargetCode
comptime SRC: Int = 0
comptime DST: Int = 1
comptime EDGE: Int = 2

# `dgl/ops/sddmm.py` :: _register_sddmm_func, op order add, sub, mul, div, dot
comptime OP_ADD: Int = 0
comptime OP_SUB: Int = 1
comptime OP_MUL: Int = 2
comptime OP_DIV: Int = 3
comptime OP_DOT: Int = 4
# `dgl/ops/spmm.py` :: _register_spmm_func, the two copy forms
comptime OP_COPY_LHS: Int = 5
comptime OP_COPY_RHS: Int = 6

# `dgl/ops/spmm.py` :: _register_spmm_func, reduce order sum, max, min, mean.
# `mean` never reaches a kernel: upstream turns it into `sum` in Python and
# divides by the in-degree, and this port does exactly the same.
comptime REDUCE_SUM: Int = 0
comptime REDUCE_MAX: Int = 1
comptime REDUCE_MIN: Int = 2
comptime REDUCE_MEAN: Int = 3

comptime NEG_INF: Float64 = -1.7976931348623157e308
comptime POS_INF: Float64 = 1.7976931348623157e308


# ---------------------------------------------------------------------------
# Reduce identity and combine, at both widths. `op`/`red` are compile-time
# parameters, so the dispatch folds away and the inner loop is a bare
# load/combine/store.
# ---------------------------------------------------------------------------
def _seed[red: Int]() -> Float64:
    if red == REDUCE_MAX:
        return NEG_INF
    elif red == REDUCE_MIN:
        return POS_INF
    return 0.0


def _seed_v[red: Int]() -> VW:
    if red == REDUCE_MAX:
        return VW(NEG_INF)
    elif red == REDUCE_MIN:
        return VW(POS_INF)
    return VW(0.0)


def _combine[op: Int, red: Int](acc: Float64, x: Float64, y: Float64) -> Float64:
    var m: Float64
    if op == OP_ADD:
        m = x + y
    elif op == OP_SUB:
        m = x - y
    elif op == OP_MUL:
        m = x * y
    elif op == OP_DIV:
        m = x / y
    elif op == OP_COPY_LHS:
        m = x
    else:
        m = y
    if red == REDUCE_MAX:
        return max(acc, m)
    elif red == REDUCE_MIN:
        return min(acc, m)
    return acc + m

def _seed_r(red: Int) -> Float64:
    if red == REDUCE_MAX:
        return NEG_INF
    elif red == REDUCE_MIN:
        return POS_INF
    return 0.0



def _combine_v[op: Int, red: Int](acc: VW, x: VW, y: VW) -> VW:
    var m: VW
    if op == OP_ADD:
        m = x + y
    elif op == OP_SUB:
        m = x - y
    elif op == OP_MUL:
        m = x * y
    elif op == OP_DIV:
        m = x / y
    elif op == OP_COPY_LHS:
        m = x
    else:
        m = y
    if red == REDUCE_MAX:
        return max(acc, m)
    elif red == REDUCE_MIN:
        return min(acc, m)
    return acc + m


# ---------------------------------------------------------------------------
# src/array/cpu/array_cumsum.cc :: array_cumsum
# ---------------------------------------------------------------------------
def cumsum(src: IP, dst: IP, n: Int) -> None:
    var acc: Int = 0
    var i = 0
    while i < n:
        acc += src.unsafe_load(i)
        dst.unsafe_store(i, acc)
        i += 1


# src/array/cpu/array_op_impl.cc :: replace_inf_with_zero
# Scalar: this dialect's SIMD comparison folds to a single Bool rather than a
# lane mask, so there is no vector select to zero one lane with.
def replace_inf_with_zero(src: FP, dst: FP, n: Int) -> None:
    var i = 0
    while i < n:
        var v = src.unsafe_load(i)
        if v == POS_INF or v == NEG_INF:
            v = 0.0
        dst.unsafe_store(i, v)
        i += 1


# src/array/cpu/array_op_impl.cc :: clamp on an integer array
def clamp_i64(src: IP, dst: IP, n: Int, lo: Int, hi: Int) -> None:
    var i = 0
    while i < n:
        var v = src.unsafe_load(i)
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        dst.unsafe_store(i, v)
        i += 1


# ---------------------------------------------------------------------------
# python/dgl/ops/sddmm.py :: gsddmm
#
#   x_e = phi(x_{lhs}, x_{rhs})  for all (u, e, v) in G
#
# `lhs_target`/`rhs_target` are TargetCode: 0 gathers by uids, 1 by vids, 2
# indexes the operand directly because it is already an edge feature. `op` is
# one of the OP_* codes. For every op but dot, `dout` equals the two operand
# widths (the Python layer enforces it after reshape_lhs_rhs); for dot the
# feature dimension is contracted and `dout` is 1.
# ---------------------------------------------------------------------------
def gsddmm(
    uids: IP,
    vids: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    num_edges: Int,
    lhs_target: Int,
    rhs_target: Int,
    op: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    vec: Bool,
) -> None:
    var e = 0
    while e < num_edges:
        # src/array/kernel/sddmm.cc gathers through a target-coded index map
        var li = 0
        if lhs_target == SRC:
            li = uids.unsafe_load(e) * dl
        elif lhs_target == DST:
            li = vids.unsafe_load(e) * dl
        else:
            li = e * dl

        var ri = 0
        if rhs_target == SRC:
            ri = uids.unsafe_load(e) * dr
        elif rhs_target == DST:
            ri = vids.unsafe_load(e) * dr
        else:
            ri = e * dr

        if op == OP_DOT:
            var vacc = VW(0.0)
            var k = 0
            if vec:
                while k + W <= dl:
                    vacc += lhs.unsafe_load[width=W](li + k) * rhs.unsafe_load[
                        width=W
                    ](ri + k)
                    k += W
            # one horizontal reduction per edge, not one per vector: the
            # shuffles it costs are what makes a per-vector reduce_add lose to
            # the scalar chain at these contraction widths
            var acc: Float64 = 0.0
            if vec:
                acc = vacc.reduce_add()
            while k < dl:
                acc += lhs.unsafe_load(li + k) * rhs.unsafe_load(ri + k)
                k += 1
            dst.unsafe_store(e, acc)
            e += 1
            continue

        var d = 0
        if vec:
            while d + W <= dout:
                var x = lhs.unsafe_load[width=W](li + d)
                var y = rhs.unsafe_load[width=W](ri + d)
                if op == OP_ADD:
                    dst.unsafe_store(e * dout + d, x + y)
                elif op == OP_SUB:
                    dst.unsafe_store(e * dout + d, x - y)
                elif op == OP_MUL:
                    dst.unsafe_store(e * dout + d, x * y)
                elif op == OP_DIV:
                    dst.unsafe_store(e * dout + d, x / y)
                elif op == OP_COPY_LHS:
                    dst.unsafe_store(e * dout + d, x)
                else:
                    dst.unsafe_store(e * dout + d, y)
                d += W
        while d < dout:
            var x = lhs.unsafe_load(li + d)
            var y = rhs.unsafe_load(ri + d)
            if op == OP_ADD:
                dst.unsafe_store(e * dout + d, x + y)
            elif op == OP_SUB:
                dst.unsafe_store(e * dout + d, x - y)
            elif op == OP_MUL:
                dst.unsafe_store(e * dout + d, x * y)
            elif op == OP_DIV:
                dst.unsafe_store(e * dout + d, x / y)
            elif op == OP_COPY_LHS:
                dst.unsafe_store(e * dout + d, x)
            else:
                dst.unsafe_store(e * dout + d, y)
            d += 1
        e += 1


# ---------------------------------------------------------------------------
# python/dgl/ops/spmm.py :: gspmm
#
#   x_v = psi_{(u,v,e) in G} ( rho(x_u, x_e) )
#
# One pass over the in-edge CSR, fused exactly as upstream's CSR SpMM kernel
# fuses it. `indptr`/`indices` are the in-edge (destination-major) CSR and
# `rhs` is indexed by the CSR row's edge, which for a destination-major CSR is
# already sorted by edge id only when the graph has no duplicate edges -- so the
# caller passes the matching edge ids in `eid`.
#
# The accumulator runs across the `dout` columns of one destination at a time,
# so a whole vector of columns is reduced per gathered row instead of one
# column at a time.
#
# Zero in-degree nodes are filled with zero, which is upstream's documented
# "replace all +/-inf values to zero" behaviour for max and min.
# ---------------------------------------------------------------------------
def _gspmm[op: Int, red: Int](
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    num_dst: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    deg: IP,
    mean: Bool,
    vec: Bool,
) -> None:
    var v = 0
    while v < num_dst:
        var start = indptr.unsafe_load(v)
        var stop = indptr.unsafe_load(v + 1)
        # `mean` is upstream's sum-then-divide, folded in here so the reduced
        # row is never written twice; the divisor is the in-degree clamped at
        # one, which is what the Python side used to hand the kernel's result
        # back to numpy to divide by.
        var scale: Float64 = 1.0
        if mean:
            scale = Float64(deg.unsafe_load(v))
            if scale < 1.0:
                scale = 1.0
        var k = 0
        if start >= stop:
            if vec:
                while k + W <= dout:
                    dst.unsafe_store(v * dout + k, VW(0.0))
                    k += W
            while k < dout:
                dst.unsafe_store(v * dout + k, 0.0)
                k += 1
            v += 1
            continue
        if vec:
            while k + W <= dout:
                var acc = _seed_v[red]()
                var i = start
                while i < stop:
                    acc = _combine_v[op, red](
                        acc,
                        lhs.unsafe_load[width=W](
                            indices.unsafe_load(i) * dl + k
                        ),
                        rhs.unsafe_load[width=W](eid.unsafe_load(i) * dr + k),
                    )
                    i += 1
                if mean:
                    acc = acc / VW(scale)
                dst.unsafe_store(v * dout + k, acc)
                k += W
        while k < dout:
            var acc = _seed[red]()
            var i = start
            while i < stop:
                acc = _combine[op, red](
                    acc,
                    lhs.unsafe_load(indices.unsafe_load(i) * dl + k),
                    rhs.unsafe_load(eid.unsafe_load(i) * dr + k),
                )
                i += 1
            if mean:
                acc = acc / scale
            dst.unsafe_store(v * dout + k, acc)
            k += 1
        v += 1


def gspmm(
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    num_dst: Int,
    op: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    reduce_op: Int,
    deg: IP,
    vec: Bool,
) -> None:
    if op == OP_ADD:
        _gspmm_op[OP_ADD](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_SUB:
        _gspmm_op[OP_SUB](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_MUL:
        _gspmm_op[OP_MUL](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_DIV:
        _gspmm_op[OP_DIV](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_COPY_LHS:
        _gspmm_op[OP_COPY_LHS](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )
    else:
        _gspmm_op[OP_COPY_RHS](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, reduce_op, deg,
            vec
        )


# One instantiation per (op, reduce) pair: the dispatch above is the whole of
# the message/reduce functor, and everything below it is a bare load, combine
# and store.
def _gspmm_op[op: Int](
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    num_dst: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    reduce_op: Int,
    deg: IP,
    vec: Bool,
) -> None:
    if reduce_op == REDUCE_MAX:
        _gspmm[op, REDUCE_MAX](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, deg,
            False, vec
        )
    elif reduce_op == REDUCE_MIN:
        _gspmm[op, REDUCE_MIN](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, deg,
            False, vec
        )
    elif reduce_op == REDUCE_MEAN:
        _gspmm[op, REDUCE_SUM](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, deg,
            True, vec
        )
    else:
        _gspmm[op, REDUCE_SUM](
            indptr, indices, eid, lhs, rhs, dst, num_dst, dl, dr, dout, deg,
            False, vec
        )


# ---------------------------------------------------------------------------
# python/dgl/ops/segment.py :: segment_reduce
#
# `offsets` is the length-(num_seg + 1) prefix sum of the segment lengths that
# `dgl/ops/segment.py` builds with F.cumsum. `sum` starts from zero; `max`/`min`
# start from the sentinels and are left for the caller to replace with zero, as
# upstream does.
# ---------------------------------------------------------------------------
def segment_reduce(
    offsets: IP,
    value: FP,
    dst: FP,
    num_seg: Int,
    d: Int,
    reduce_op: Int,
    n: Int,
    vec: Bool,
) -> None:
    var s = 0
    while s < num_seg:
        var start = offsets.unsafe_load(s)
        var stop = offsets.unsafe_load(s + 1)
        # A descending or out-of-range offsets array would otherwise let the
        # vectoriser read past the end of `value`; `n` is the row count of
        # `value`, so clamp both ends into it.
        if start < 0:
            start = 0
        if stop > n:
            stop = n
        if stop < start:
            stop = start
        var k = 0
        if vec:
            while k + W <= d:
                if reduce_op == REDUCE_SUM:
                    var acc = VW(0.0)
                    var i = start
                    while i < stop:
                        acc += value.unsafe_load[width=W](i * d + k)
                        i += 1
                    dst.unsafe_store(s * d + k, acc)
                elif reduce_op == REDUCE_MAX:
                    var acc = VW(NEG_INF)
                    var i = start
                    while i < stop:
                        acc = max(acc, value.unsafe_load[width=W](i * d + k))
                        i += 1
                    dst.unsafe_store(s * d + k, acc)
                else:
                    var acc = VW(POS_INF)
                    var i = start
                    while i < stop:
                        acc = min(acc, value.unsafe_load[width=W](i * d + k))
                        i += 1
                    dst.unsafe_store(s * d + k, acc)
                k += W
        while k < d:
            var acc = _seed_r(reduce_op)
            var i = start
            while i < stop:
                var x = value.unsafe_load(i * d + k)
                if reduce_op == REDUCE_MAX:
                    acc = max(acc, x)
                elif reduce_op == REDUCE_MIN:
                    acc = min(acc, x)
                else:
                    acc += x
                i += 1
            dst.unsafe_store(s * d + k, acc)
            k += 1
        s += 1


# python/dgl/ops/segment.py :: segment_softmax
#
#   out[i] = exp(value[i] - max) / sum_{j in seg} exp(value[j] - max)
#
# The five passes and two temporaries that upstream's Python composition pays
# for (segment max, repeat, subtract, exp, segment sum, repeat, divide) are
# these three passes over the same rows: the per-column max and the running
# exponential sum live in `scratch`, two `d`-long rows the caller owns. The
# arithmetic and its order are upstream's: the same values are summed in the
# same row order, so the result is the composition's result.
def segment_softmax(
    offsets: IP, value: FP, dst: FP, scratch: FP, num_seg: Int, d: Int, n: Int, vec: Bool
) -> None:
    var s = 0
    while s < num_seg:
        var start = offsets.unsafe_load(s)
        var stop = offsets.unsafe_load(s + 1)
        if start < 0:
            start = 0
        if stop > n:
            stop = n
        if stop < start:
            stop = start
        # pass 1: the segment's per-column max, `scratch[0:d]`
        var k = 0
        while k < d:
            scratch.unsafe_store(k, NEG_INF)
            k += 1
        var i = start
        while i < stop:
            k = 0
            if vec:
                while k + W <= d:
                    scratch.unsafe_store(
                        k,
                        max(
                            scratch.unsafe_load[width=W](k),
                            value.unsafe_load[width=W](i * d + k),
                        ),
                    )
                    k += W
            while k < d:
                scratch.unsafe_store(
                    k, max(scratch.unsafe_load(k), value.unsafe_load(i * d + k))
                )
                k += 1
            i += 1
        # pass 2: exp(value - max) into `dst`, running sum into `scratch[d:2d]`
        k = 0
        while k < d:
            scratch.unsafe_store(d + k, 1.0 if start == stop else 0.0)
            k += 1
        i = start
        while i < stop:
            k = 0
            if vec:
                while k + W <= d:
                    var e = exp(
                        value.unsafe_load[width=W](i * d + k)
                        - scratch.unsafe_load[width=W](k)
                    )
                    dst.unsafe_store(i * d + k, e)
                    scratch.unsafe_store(
                        d + k, scratch.unsafe_load[width=W](d + k) + e
                    )
                    k += W
            while k < d:
                var e = exp(value.unsafe_load(i * d + k) - scratch.unsafe_load(k))
                dst.unsafe_store(i * d + k, e)
                scratch.unsafe_store(d + k, scratch.unsafe_load(d + k) + e)
                k += 1
            i += 1
        # pass 3: divide by the segment's exponential sum
        i = start
        while i < stop:
            k = 0
            if vec:
                while k + W <= d:
                    dst.unsafe_store(
                        i * d + k,
                        dst.unsafe_load[width=W](i * d + k)
                        / scratch.unsafe_load[width=W](d + k),
                    )
                    k += W
            while k < d:
                dst.unsafe_store(
                    i * d + k,
                    dst.unsafe_load(i * d + k) / scratch.unsafe_load(d + k),
                )
                k += 1
            i += 1
        s += 1


# python/dgl/ops/segment.py :: segment_mm
#   out[i] = a[i] @ b[r], where r is the segment i belongs to
def segment_mm(
    a: FP, b: FP, dst: FP, seglen_a: IP, n: Int, d1: Int, r: Int, d2: Int, vec: Bool
) -> None:
    var row = 0
    var seg = 0
    while seg < r:
        var end = row + seglen_a.unsafe_load(seg)
        while row < end:
            var j = 0
            if vec:
                while j + W <= d2:
                    var vacc = VW(0.0)
                    var k = 0
                    while k < d1:
                        vacc += a.unsafe_load(row * d1 + k) * b.unsafe_load[
                            width=W
                        ]((seg * d1 + k) * d2 + j)
                        k += 1
                    dst.unsafe_store(row * d2 + j, vacc.reduce_add())
                    j += W
            while j < d2:
                var acc: Float64 = 0.0
                var k = 0
                while k < d1:
                    acc += a.unsafe_load(row * d1 + k) * b.unsafe_load(
                        (seg * d1 + k) * d2 + j
                    )
                    k += 1
                dst.unsafe_store(row * d2 + j, acc)
                j += 1
            row += 1
        seg += 1


# ---------------------------------------------------------------------------
# python/dgl/ops/edge_softmax.py :: edge_softmax
#
#   a_ij = exp(z_ij) / sum_{j in N(i)} exp(z_ij)
#
# The CSR is the in-edge CSR for norm_by="dst" and the out-edge CSR for
# norm_by="src"; `eid` maps a CSR row back to the edge id used by the caller, and
# `mask` (all ones when eids is ALL) selects which edges take part. Max is
# subtracted before the exponential, as in upstream's fused kernel. `all_ones`
# says the mask is the all-ones selection, which lets the per-edge select
# disappear from all three passes.
# ---------------------------------------------------------------------------
def edge_softmax_(
    indptr: IP,
    indices: IP,
    eid: IP,
    mask: BP,
    logits: FP,
    dst: FP,
    num_nodes: Int,
    all_ones: Int,
) -> None:
    var i = 0
    while i < num_nodes:
        var start = indptr.unsafe_load(i)
        var stop = indptr.unsafe_load(i + 1)
        var m: Float64 = NEG_INF
        var any = False
        var j = start
        if all_ones == 1:
            while j < stop:
                m = max(m, logits.unsafe_load(eid.unsafe_load(j)))
                any = True
                j += 1
        else:
            while j < stop:
                var e = eid.unsafe_load(j)
                if mask.unsafe_load(e) == 1:
                    m = max(m, logits.unsafe_load(e))
                    any = True
                j += 1
        if not any:
            j = start
            while j < stop:
                var e = eid.unsafe_load(j)
                if all_ones == 1 or mask.unsafe_load(e) == 1:
                    dst.unsafe_store(e, 0.0)
                j += 1
            i += 1
            continue
        var acc: Float64 = 0.0
        j = start
        if all_ones == 1:
            while j < stop:
                var e = eid.unsafe_load(j)
                var v = exp(logits.unsafe_load(e) - m)
                dst.unsafe_store(e, v)
                acc += v
                j += 1
            j = start
            while j < stop:
                var e = eid.unsafe_load(j)
                dst.unsafe_store(e, dst.unsafe_load(e) / acc)
                j += 1
        else:
            while j < stop:
                var e = eid.unsafe_load(j)
                if mask.unsafe_load(e) == 1:
                    var v = exp(logits.unsafe_load(e) - m)
                    dst.unsafe_store(e, v)
                    acc += v
                j += 1
            j = start
            while j < stop:
                var e = eid.unsafe_load(j)
                if mask.unsafe_load(e) == 1:
                    dst.unsafe_store(e, dst.unsafe_load(e) / acc)
                j += 1
        i += 1


# ---------------------------------------------------------------------------
# python/dgl/ops/gather_mm.py :: gather_mm
#   c[i] = a[i] @ b[idx_b[i]]
# ---------------------------------------------------------------------------
def gather_mm(a: FP, b: FP, idx_b: IP, dst: FP, n: Int, d1: Int, d2: Int) -> None:
    var i = 0
    while i < n:
        var o = idx_b.unsafe_load(i) * d1 * d2
        var j = 0
        while j < d2:
            var acc: Float64 = 0.0
            var k = 0
            while k < d1:
                acc += a.unsafe_load(i * d1 + k) * b.unsafe_load(o + k * d2 + j)
                k += 1
            dst.unsafe_store(i * d2 + j, acc)
            j += 1
        i += 1
