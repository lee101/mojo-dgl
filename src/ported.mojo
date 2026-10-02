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

from max.algorithm import parallelize
from std.math import exp, inf
from std.sys import simd_width_of

comptime FP = Pointer[Float64, AnyOrigin[mut=True]]
comptime IP = Pointer[Int, AnyOrigin[mut=True]]
comptime BP = Pointer[Int8, AnyOrigin[mut=True]]

comptime W = simd_width_of[DType.float64]()
comptime VW = SIMD[DType.float64, W]

# Worker count and the work threshold below which a kernel stays serial. Every
# feature kernel here partitions over its own output axis -- destination node,
# edge, node, segment, row -- and no output element reads another, so a chunk
# boundary changes nothing about the value computed. That is why the split is
# exact rather than approximate. Python reads `WORKERS` back through the
# `mdgl_workers` export to size the per-worker scratch rows.
comptime WORKERS: Int = 32
# Below this many output elements the launch itself costs more than the work.
comptime PARALLEL_MIN: Int = 4096

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

comptime NEG_INF: Float64 = -inf[DType.float64]()
comptime POS_INF: Float64 = inf[DType.float64]()


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
    # Element i depends only on element i, so the split over `[i0, i1)` is exact.
    if n >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var i0 = n * task // WORKERS
            var i1 = n * (task + 1) // WORKERS
            _riwz_range(src, dst, i0, i1)

        parallelize(work, WORKERS, WORKERS)
    else:
        _riwz_range(src, dst, 0, n)


def _riwz_range(src: FP, dst: FP, i0: Int, i1: Int) -> None:
    var i = i0
    while i < i1:
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
    # Each edge writes one output row and reads only the two gathered operand
    # rows, so the split over `[e0, e1)` is exact.
    if num_edges >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var e0 = num_edges * task // WORKERS
            var e1 = num_edges * (task + 1) // WORKERS
            _gsddmm_range(
                uids, vids, lhs, rhs, dst, e0, e1, lhs_target, rhs_target, op, dl,
                dr, dout, vec
            )

        parallelize(work, WORKERS, WORKERS)
    else:
        _gsddmm_range(
            uids, vids, lhs, rhs, dst, 0, num_edges, lhs_target, rhs_target, op, dl,
            dr, dout, vec
        )


def _gsddmm_range(
    uids: IP,
    vids: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    e0: Int,
    e1: Int,
    lhs_target: Int,
    rhs_target: Int,
    op: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    vec: Bool,
) -> None:
    var e = e0
    while e < e1:
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
# A node with no in-edges keeps the reduce identity: zero for sum, -inf for
# max, +inf for min. That is what `dgl.ops.gspmm` returns, and it is
# `update_all` — not this kernel — that then replaces those with zero, as
# upstream's message-passing VM does.
# ---------------------------------------------------------------------------
def _gspmm[op: Int, red: Int](
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    v0: Int,
    v1: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    deg: IP,
    mean: Bool,
    vec: Bool,
) -> None:
    var v = v0
    while v < v1:
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
                    dst.unsafe_store(v * dout + k, _seed_v[red]())
                    k += W
            while k < dout:
                dst.unsafe_store(v * dout + k, _seed[red]())
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
    # Destination rows are independent, so the split is over `[v0, v1)` and a
    # chunk boundary does not change any output element.
    if num_dst >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var v0 = num_dst * task // WORKERS
            var v1 = num_dst * (task + 1) // WORKERS
            _gspmm_dispatch(
                indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, op,
                reduce_op, deg, vec
            )

        parallelize(work, WORKERS, WORKERS)
    else:
        _gspmm_dispatch(
            indptr, indices, eid, lhs, rhs, dst, 0, num_dst, dl, dr, dout, op,
            reduce_op, deg, vec
        )


# One instantiation per (op, reduce) pair: the dispatch below is the whole of
# the message/reduce functor, and everything under it is a bare load, combine
# and store.
def _gspmm_dispatch(
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    v0: Int,
    v1: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    op: Int,
    reduce_op: Int,
    deg: IP,
    vec: Bool,
) -> None:
    if op == OP_ADD:
        _gspmm_op[OP_ADD](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_SUB:
        _gspmm_op[OP_SUB](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_MUL:
        _gspmm_op[OP_MUL](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_DIV:
        _gspmm_op[OP_DIV](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )
    elif op == OP_COPY_LHS:
        _gspmm_op[OP_COPY_LHS](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )
    else:
        _gspmm_op[OP_COPY_RHS](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, reduce_op, deg,
            vec
        )


def _gspmm_op[op: Int](
    indptr: IP,
    indices: IP,
    eid: IP,
    lhs: FP,
    rhs: FP,
    dst: FP,
    v0: Int,
    v1: Int,
    dl: Int,
    dr: Int,
    dout: Int,
    reduce_op: Int,
    deg: IP,
    vec: Bool,
) -> None:
    if reduce_op == REDUCE_MAX:
        _gspmm[op, REDUCE_MAX](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, deg,
            False, vec
        )
    elif reduce_op == REDUCE_MIN:
        _gspmm[op, REDUCE_MIN](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, deg,
            False, vec
        )
    elif reduce_op == REDUCE_MEAN:
        _gspmm[op, REDUCE_SUM](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, deg,
            True, vec
        )
    else:
        _gspmm[op, REDUCE_SUM](
            indptr, indices, eid, lhs, rhs, dst, v0, v1, dl, dr, dout, deg,
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
    # Each segment writes one output row and reads only its own input rows, so
    # the split over `[s0, s1)` is exact.
    if num_seg >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var s0 = num_seg * task // WORKERS
            var s1 = num_seg * (task + 1) // WORKERS
            _seg_reduce_range(
                offsets, value, dst, s0, s1, d, reduce_op, n, vec
            )

        parallelize(work, WORKERS, WORKERS)
    else:
        _seg_reduce_range(offsets, value, dst, 0, num_seg, d, reduce_op, n, vec)


def _seg_reduce_range(
    offsets: IP,
    value: FP,
    dst: FP,
    s0: Int,
    s1: Int,
    d: Int,
    reduce_op: Int,
    n: Int,
    vec: Bool,
) -> None:
    var s = s0
    while s < s1:
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
    # Each segment reads and writes only its own rows, and each worker gets its
    # own pair of `d`-long scratch rows, so the split over `[s0, s1)` is exact.
    if num_seg >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var s0 = num_seg * task // WORKERS
            var s1 = num_seg * (task + 1) // WORKERS
            _seg_softmax_range(
                offsets, value, dst, scratch.unsafe_offset(task * 2 * d), s0, s1, d,
                n, vec
            )

        parallelize(work, WORKERS, WORKERS)
    else:
        _seg_softmax_range(offsets, value, dst, scratch, 0, num_seg, d, n, vec)


def _seg_softmax_range(
    offsets: IP,
    value: FP,
    dst: FP,
    scratch: FP,
    s0: Int,
    s1: Int,
    d: Int,
    n: Int,
    vec: Bool,
) -> None:
    var s = s0
    while s < s1:
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
                        vacc += VW(a.unsafe_load(row * d1 + k)) * b.unsafe_load[
                            width=W
                        ]((seg * d1 + k) * d2 + j)
                        k += 1
                    dst.unsafe_store[width=W](row * d2 + j, vacc)
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

# Upstream's logits are (E, *, 1): every trailing column past the edge axis is
# one independent softmax, which is how a multi-head attention layer calls it.
# The node is the outer loop and the head is the inner one, so one walk of a
# node's CSR row covers every head instead of one walk per head; the row's
# running max and sum live in `scratch`, two `d`-long rows the caller owns and
# only one node needs at a time. A head is `W` lanes at a time, with a scalar
# tail for the `d % W` columns that do not fill a vector, so the per-edge body
# is a contiguous vector of the row rather than a scalar gather.
# ---------------------------------------------------------------------------
def _es_max_row(logits: FP, o: Int, mx: FP, sb: Int, d: Int) -> None:
    var h = 0
    while h + W <= d:
        mx.unsafe_store(sb + h, max(mx.unsafe_load[width=W](sb + h), logits.unsafe_load[width=W](o + h)))
        h += W
    while h < d:
        mx.unsafe_store(sb + h, max(mx.unsafe_load(sb + h), logits.unsafe_load(o + h)))
        h += 1


def _es_exp_row(logits: FP, dst: FP, o: Int, scratch: FP, sb: Int, d: Int) -> None:
    var h = 0
    while h + W <= d:
        var v = exp(
            logits.unsafe_load[width=W](o + h) - scratch.unsafe_load[width=W](sb + h)
        )
        dst.unsafe_store(o + h, v)
        scratch.unsafe_store(sb + d + h, scratch.unsafe_load[width=W](sb + d + h) + v)
        h += W
    while h < d:
        var v = exp(logits.unsafe_load(o + h) - scratch.unsafe_load(sb + h))
        dst.unsafe_store(o + h, v)
        scratch.unsafe_store(sb + d + h, scratch.unsafe_load(sb + d + h) + v)
        h += 1


def _es_div_row(dst: FP, o: Int, scratch: FP, sb: Int, d: Int) -> None:
    var h = 0
    while h + W <= d:
        dst.unsafe_store(
            o + h, dst.unsafe_load[width=W](o + h) / scratch.unsafe_load[width=W](sb + d + h)
        )
        h += W
    while h < d:
        dst.unsafe_store(o + h, dst.unsafe_load(o + h) / scratch.unsafe_load(sb + d + h))
        h += 1


def _es_zero_row(dst: FP, o: Int, d: Int) -> None:
    var h = 0
    while h + W <= d:
        dst.unsafe_store(o + h, VW(0.0))
        h += W
    while h < d:
        dst.unsafe_store(o + h, 0.0)
        h += 1


def edge_softmax_(
    indptr: IP,
    indices: IP,
    eid: IP,
    mask: BP,
    logits: FP,
    dst: FP,
    scratch: FP,
    num_nodes: Int,
    all_ones: Int,
    d: Int,
) -> None:
    # `d == 1` keeps its own scalar kernel: with a single column the running max
    # and sum stay in registers, and routing it through scratch measured slower.
    if d == 1:
        _es_one_head(indptr, eid, mask, logits, dst, num_nodes, all_ones)
        return
    # Nodes are independent, and each worker gets its own pair of `d`-long
    # scratch rows, so the split over `[i0, i1)` is exact.
    if num_nodes >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var i0 = num_nodes * task // WORKERS
            var i1 = num_nodes * (task + 1) // WORKERS
            _es_range(
                indptr,
                eid,
                mask,
                logits,
                dst,
                scratch.unsafe_offset(task * 2 * d),
                i0,
                i1,
                all_ones, d
            )

        parallelize(work, WORKERS, WORKERS)
    else:
        _es_range(
            indptr, eid, mask, logits, dst, scratch, 0, num_nodes, all_ones, d
        )


def _es_range(
    indptr: IP,
    eid: IP,
    mask: BP,
    logits: FP,
    dst: FP,
    scratch: FP,
    i0: Int,
    i1: Int,
    all_ones: Int,
    d: Int,
) -> None:
    var i = i0
    while i < i1:
        var start = indptr.unsafe_load(i)
        var stop = indptr.unsafe_load(i + 1)
        if start < stop:
            var h = 0
            while h < d:
                scratch.unsafe_store(h, NEG_INF)
                h += 1
            var any = False
            var j = start
            if all_ones == 1:
                while j < stop:
                    _es_max_row(logits, eid.unsafe_load(j) * d, scratch, 0, d)
                    any = True
                    j += 1
            else:
                while j < stop:
                    var e = eid.unsafe_load(j)
                    if mask.unsafe_load(e) == 1:
                        _es_max_row(logits, e * d, scratch, 0, d)
                        any = True
                    j += 1
            if any:
                h = 0
                while h < d:
                    scratch.unsafe_store(d + h, 0.0)
                    h += 1
                j = start
                if all_ones == 1:
                    while j < stop:
                        _es_exp_row(logits, dst, eid.unsafe_load(j) * d, scratch, 0, d)
                        j += 1
                else:
                    while j < stop:
                        var e = eid.unsafe_load(j)
                        if mask.unsafe_load(e) == 1:
                            _es_exp_row(logits, dst, e * d, scratch, 0, d)
                        j += 1
                j = start
                if all_ones == 1:
                    while j < stop:
                        _es_div_row(dst, eid.unsafe_load(j) * d, scratch, 0, d)
                        j += 1
                else:
                    while j < stop:
                        var e = eid.unsafe_load(j)
                        if mask.unsafe_load(e) == 1:
                            _es_div_row(dst, e * d, scratch, 0, d)
                        j += 1
            else:
                j = start
                while j < stop:
                    var e = eid.unsafe_load(j)
                    if all_ones == 1 or mask.unsafe_load(e) == 1:
                        _es_zero_row(dst, e * d, d)
                    j += 1
        i += 1


# The single-column form keeps the running max and sum in registers instead of
# in the scratch rows: with one column there is nothing for a `d`-long scratch
# row to carry, and routing it through scratch measured slower. Nodes are still
# independent, so it splits over nodes the same way.
def _es_one_head(
    indptr: IP,
    eid: IP,
    mask: BP,
    logits: FP,
    dst: FP,
    num_nodes: Int,
    all_ones: Int,
) -> None:
    if num_nodes >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var i0 = num_nodes * task // WORKERS
            var i1 = num_nodes * (task + 1) // WORKERS
            _es_one_head_range(indptr, eid, mask, logits, dst, i0, i1, all_ones)

        parallelize(work, WORKERS, WORKERS)
    else:
        _es_one_head_range(indptr, eid, mask, logits, dst, 0, num_nodes, all_ones)


def _es_one_head_range(
    indptr: IP,
    eid: IP,
    mask: BP,
    logits: FP,
    dst: FP,
    i0: Int,
    i1: Int,
    all_ones: Int,
) -> None:
    var i = i0
    while i < i1:
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


# The contraction is over `k` and the output is over `j`, so a `W`-wide load
# takes `W` adjacent output columns of one `b` row at a time and accumulates
# them as `W` partial sums, which are then stored as a vector. The `b` block a
# row walks is `d1 * d2` doubles and stays in L1 across the whole row, so this
# touches the same memory the scalar form did with a quarter of the loads.
# ---------------------------------------------------------------------------
def gather_mm(a: FP, b: FP, idx_b: IP, dst: FP, n: Int, d1: Int, d2: Int) -> None:
    # Each row writes its own output block and reads only its own `a` row and
    # its own `b` block, so the split over `[i0, i1)` is exact.
    if n >= PARALLEL_MIN:
        def work(task: Int) {`imm`}:
            var i0 = n * task // WORKERS
            var i1 = n * (task + 1) // WORKERS
            _gather_mm_range(a, b, idx_b, dst, i0, i1, d1, d2)

        parallelize(work, WORKERS, WORKERS)
    else:
        _gather_mm_range(a, b, idx_b, dst, 0, n, d1, d2)


def _gather_mm_range(
    a: FP, b: FP, idx_b: IP, dst: FP, i0: Int, i1: Int, d1: Int, d2: Int
) -> None:
    var i = i0
    while i < i1:
        var o = idx_b.unsafe_load(i) * d1 * d2
        var arow = i * d1
        var drow = i * d2
        var j = 0
        while j + W <= d2:
            var vacc = VW(0.0)
            var k = 0
            while k < d1:
                vacc += VW(a.unsafe_load(arow + k)) * b.unsafe_load[width=W](
                    o + k * d2 + j
                )
                k += 1
            dst.unsafe_store[width=W](drow + j, vacc)
            j += W
        while j < d2:
            var acc: Float64 = 0.0
            var k = 0
            while k < d1:
                acc += a.unsafe_load(arow + k) * b.unsafe_load(o + k * d2 + j)
                k += 1
            dst.unsafe_store(drow + j, acc)
            j += 1
        i += 1
