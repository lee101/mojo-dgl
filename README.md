# mojo-dgl

[DGL](https://dmlc.github.io/dgl/)'s graph message-passing kernels, ported to
[Mojo](https://www.modular.com/mojo) and callable from Python under DGL's own
names. `dgl.graph`, `dgl.ops`, `dgl.function`, `g.update_all(fn.u_mul_e('h',
'e', 'm'), fn.sum('m', 'h'))` all mean what they mean upstream, so for the
covered subset changing the import is usually the whole migration.

```python
import numpy as np
import mojodgl as dgl
import mojodgl.function as fn

u = np.array([0, 1, 2, 0, 1])
v = np.array([1, 2, 0, 2, 2])

g = dgl.graph((u, v))
g.ndata["h"] = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
g.edata["e"] = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])

g.update_all(fn.u_mul_e("h", "e", "m"), fn.mean("m", "z"))
print(g.ndata["z"])

```text
[[0.         2.        ]
 [0.5        0.        ]
 [0.66666667 0.33333333]]
```

Upstream spells the package `dgl`; the port is `mojodgl` so it can sit next to
the real thing in the same environment, which is exactly how the test suite runs
— every parity test calls the same function name on both libraries.

Every function below is checked against real upstream `dgl` 2.3.0 on the same
arrays, plus the worked examples printed in upstream's own docstrings.

## What is implemented

| upstream module | what is here |
| --- | --- |
| `dgl.convert` | `graph(data, *, num_nodes=None, idtype=None, device=None, row_sorted=False, col_sorted=False)` |
| `dgl.heterograph` | `DGLGraph`: `number_of_nodes/src_nodes/dst_nodes`, `num_nodes/num_edges`, `number_of_edges`, `edges`, `in_edges`, `out_edges`, `in_degrees`, `out_degrees`, `canonical_etypes`, `ndata`, `edata`, `update_all`, `pull` |
| `dgl.function` | `TargetCode`, `BuiltinFunction`, `BinaryMessageFunction`, `CopyMessageFunction`, `copy_u`, `copy_e`, all 30 generated `u_add_v` … `e_dot_v` message builtins, `sum` / `max` / `min` / `mean` |
| `dgl.ops.sddmm` | `gsddmm`, `copy_u`, `copy_v`, `copy_e`, all 30 generated `u_add_v` … `e_dot_v` SDDMMs |
| `dgl.ops.spmm` | `gspmm` and all 24 generated `u_{add,sub,mul,div}_e_{sum,max,min,mean}`, `copy_u_*`, `copy_e_*` |
| `dgl.ops.segment` | `segment_reduce`, `segment_softmax`, `segment_mm` |
| `dgl.ops.edge_softmax` | `edge_softmax(graph, logits, eids=ALL, norm_by="dst")`, on `(E,)`, `(E, 1)` and upstream's `(E, *, 1)` |
| `dgl.ops.gather_mm` | `gather_mm(a, b, *, idx_b)`, both the fused and the segment-`mm` path |
| `dgl.nn.functional` | `edge_softmax` (upstream 2.x re-exports it from `dgl.ops`) |

That is 64 names in `mojodgl.ops`: 34 from `sddmm`, 25 from `spmm`, and the 5
from `segment`, `edge_softmax` and `gather_mm`. On the Mojo side
`src/ported.mojo` is the kernel file, laid out in upstream's own source order:
the array primitives first (they are what `dgl/ops/segment.py` is written
against), then `dgl/function`, then `dgl/ops` in the order `ops/__init__.py`
imports it.

`reshape_lhs_rhs` is ported and reachable as `mojodgl.ops.sddmm.reshape_lhs_rhs`,
but it is not in `ops.__all__` — upstream does not export it either. The base
classes `MessageFunction`, `ReduceFunction` and `SimpleReduceFunction` exist in
this port (they are what the builtins subclass) but, as upstream, they are not
re-exported from `mojodgl.function`; import them from
`mojodgl.function.message` and `mojodgl.function.reducer`.

## What is not implemented, and why

Every item below is a deliberate limit, not an oversight.

- **Heterographs.** `dgl.heterograph.DGLHeteroGraph`, `dgl.heterograph` and
  `dgl.hg_to_dgl` are not ported; `DGLGraph` here is homogeneous only. Every
  kernel underneath (`gsddmm`, `gspmm`, segment reduce, edge softmax) is
  per-relation, so a heterograph is a loop over the per-relation calls, but that
  loop is upstream's C++ and it is not in this pass.
- **`dgl.sparse`.** `SparseMatrix`, `SpMM`, `SDDMM`, `softmax`, `Reduction` and
  the rest are `torch.classes` objects backed by `libdgl_sparse_pytorch`; there
  is no numpy-shaped kernel to port. The arithmetic they perform is covered by
  `dgl.ops.gsddmm` / `dgl.ops.gspmm`, which are ported.
- **Message-function interpretation.** Upstream's `BuiltinFunction` objects are
  interpreted by a C++ message-passing VM. Here they are the same Python
  classes with the same fields, and `DGLGraph.update_all` dispatches them
  straight to the fused kernel that computes the same arithmetic — the
  `gspmm` path when the message is `u_op_e`, the `gsddmm` + segment-reduce path
  otherwise.
- **The mailbox handed to a reduce UDF.** Upstream passes a ragged
  `(N, max_in_degree, D)` mailbox; this port passes the `(N, D)` sum over
  in-edges, zero-filled for nodes with no in-edges. The builtin reducers are
  unaffected — they go through the kernel, not the mailbox.
- **`edge_softmax` with a subset `eids`.** Upstream's kernel is handed a CSR
  rebuilt from the selection, and it indexes `logits` by POSITION in `eids`
  rather than by edge id: `Edge_softmax_forward` calls `CalcBcastOff` and then
  `Edge_softmax_csr_forward` on a `CSRMatrix` whose row length is the number of
  selected edges sharing that endpoint, so row `i` reads `logits[i]`, not
  `logits[eids[i]]`. This port reproduces it exactly rather than "fixing" it:
  the selection's positions are grouped by endpoint, in the order they appear in
  `eids`, and position `i` takes the softmax of `logits[i]` over its group. Both
  `norm_by` share the behaviour, and with `eids` is `ALL` positions and edge
  ids coincide, so the `ALL` path is unaffected. The rule was established by
  black-box probing and is pinned by `tests/test_edge_softmax_eids.py`, which
  checks it against upstream at selection sizes 1 to 16 for both `norm_by`,
  states it as an explicit rule, and re-derives it from 200 random selections.
  Upstream's own docstring example, which shows a four-row result for a
  four-edge selection, is stale relative to the shipped kernel: it returns a
  buffer as long as `logits`, with the selected results in the first
  `len(eids)` positions, and so does this port. Logits themselves follow
  upstream's documented `(E, *, 1)` shape: `(E,)`, `(E, 1)` and `(E, H, 1)` all
  work, and every trailing column is an independent softmax.
- **`pull`.** It runs the same message-passing machinery as `update_all` over
  the induced in-edge subgraph, with builtin or user-defined message functions,
  and writes back only the requested rows.
- **Not ported at all:** `send`, `recv`, `send_and_recv`, `push`, `dgl.readout`,
  `dgl.dataloading`, `dgl.sampling`, `dgl.transforms`, `dgl.nn.pytorch` and
  every other framework binding, autograd/backward for any of the above, and the
  GPU path (this toolchain has no `std.gpu` at all, so there is no
  `DeviceContext` to launch one with; see the Performance section).
- **float64 only.** Upstream propagates whatever dtype the operands carry. The
  kernels are `Float64`; `ndata`/`edata` convert on assignment.
- **One forced rename.** `out` is a reserved word in Mojo and cannot be a
  parameter name, so the output buffer is called `dst` throughout
  `src/ported.mojo`. That is the only place a kernel signature departs from the
  shape of its upstream counterpart; the Python-facing names, argument order and
  defaults are untouched.

## Install

```bash
pixi install       # brings its own Mojo toolchain and upstream dgl
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-dgl.so
pixi run test      # parity tests against upstream dgl
pixi run bench
```

`pip install .` also works and ships the same Python package, but the compiled
library is not in the wheel: the first import builds it with `mojo`, or with
whatever `MOJODGL_MOJO` points at. `pixi install && pixi run build` is the
supported path, because it pins the toolchain the kernels are written against.

`pixi run test` and `pixi run bench` first run `build/install_upstream_deps.py`,
which installs `torchdata==0.7.1` from PyPI. dgl 2.3.0 predates torchdata 0.10,
which removed `torchdata.datapipes`; the conda solver happily resolves dgl's
own `torchdata >=0.5.0` to 0.11, and `import dgl` then dies inside
`dgl.graphbolt.base`. The script is a no-op once `torchdata.datapipes` imports,
and it warns rather than failing if it cannot fix it, in which case the tests
that need dgl skip.

## Performance

The table below is the median of four `pixi run bench-here` invocations in this
session (the flock-wrapped form of `pixi run bench`; see `bench/verify.md` for
why `pixi run bench` alone prints only a guard) on one graph of 200,000 nodes
and 1,000,000 random edges, float64 features, 32 columns. Both libraries get
the same arrays, each number is the best of three runs within an invocation,
and the table reports the median across the four invocations in this session.
Nothing here is extrapolated.
Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores, Linux x86_64,
AVX2, torch 2.3.1.post100, dgl 2.3.0. `W`, the SIMD width these kernels are
built at, is 4 float64 lanes here.

| case | mojo-dgl | dgl | speedup |
| --- | ---: | ---: | ---: |
| `u_mul_e_sum` (1M edges, 32d) | 30.5 ms | 30.2 ms | 1.01x |
| `u_add_e_mean` (1M edges, 32d) | 39.6 ms | 39.8 ms | 1.02x |
| `u_div_e_min` (1M edges, 32d) | 36.8 ms | 182.2 ms | 5.29x |
| `copy_u_max` (1M edges, 32d) | 22.6 ms | 40.3 ms | 1.89x |
| `copy_e_mean` (1M edges, 32d) | 26.2 ms | 35.6 ms | 1.29x |
| `u_dot_e` (SDDMM, 1M edges, 32d) | 22.1 ms | 10.7 ms | 0.49x |
| `u_mul_v` (SDDMM, 1M edges, 32d) | 45.0 ms | 50.2 ms | 1.14x |
| `edge_softmax` norm_by=dst (1M edges) | 11.9 ms | 2.4 ms | 0.22x |
| `edge_softmax` norm_by=src (1M edges) | 12.2 ms | 2.7 ms | 0.23x |
| `edge_softmax` 8 heads (1M edges, `(E, 8, 1)`) | 29.9 ms | 22.9 ms | 0.77x |
| `segment_reduce` sum (1M rows, 32d, 200k segments) | 18.8 ms | 21.0 ms | 1.07x |
| `segment_reduce` max (1M rows, 32d, 200k segments) | 36.7 ms | 72.9 ms | 1.96x |
| `segment_softmax` (1M rows, 32d, 200k segments) | 65.8 ms | 399.6 ms | **6.27x** |
| `gather_mm` (1M rows, 16x16, 64 blocks) | 28.0 ms | 169.6 ms | **7.09x** |
| `update_all` `u_mul_e` + `sum` (1M edges, 32d) | 29.4 ms | 27.5 ms | 0.97x |
| `update_all` `copy_u` + `max` (1M edges, 32d) | 36.8 ms | 71.8 ms | 2.17x |

**Read the two columns apart before reading the ratio.** The box this was
measured on is shared and heavily loaded (load averages observed 9–77 over the
verify runs), which penalises the two columns unequally: this port is 32
threads and dgl's kernels are OpenMP over every core, so a loaded box distorts
them by different amounts. The ratios above are therefore an upper bound on
this port's standing, not a clean measurement of either library at its best.
Nothing in the table is extrapolated: every number is the best of three runs
of the case named within an invocation, medianed over the four verify
invocations recorded in `bench/verify.md`.

mojodgl loses five rows of this table outright — `u_dot_e`, `edge_softmax`
(dst, src, and 8 heads), and `update_all` `u_mul_e` + `sum` — and is at
parity within noise on `u_mul_e_sum` (1.01x, 0.85x–1.17x across this session's
four invocations) and `segment_reduce` sum (1.07x, 0.95x–1.13x). The reason is
that dgl's kernels are multi-threaded C++ while the work these kernels do is
memory-bound. The ranking inside the table is worth reading. The wins —
`gather_mm`, `segment_softmax`, `u_div_e_min`, `copy_u_max`, `segment_reduce`
max, `update_all` `copy_u` + `max` — are the cases that are one sequential
scan over memory, or a matmul small enough to stay in cache, with a cheap
inner body. That is where upstream's threads have least to add. `u_mul_e_sum`
and `update_all` `u_mul_e` + `sum` swing across 1.0x between individual
invocations in this session (see `bench/verify.md`), i.e. they are at parity
within run-to-run noise on this box, not a clear win or loss.

The worst cases are the ones whose cost is memory latency. `u_dot_e` and
`u_mul_v` issue two or more random-indexed loads per edge, and that latency is
hard to hide. Both kernels are already moving hundreds of megabytes in the time
shown, which is the machine's random-gather bandwidth, so there is little
arithmetic left to take out of either — see "What was tried and did not pay"
below.

### Where the time goes in this port

- **The column loop is a vector loop.** `gspmm`, `gsddmm`, `segment_reduce`,
  `segment_softmax` and `segment_mm` walk the feature columns `W` lanes at a
  time (`p.unsafe_load[width=W]`, `p.unsafe_store`, free `min`/`max`, which
  work on `SIMD`), with a scalar tail for the `d % W` columns that do not fill
  a vector. `capi.mojo` sets the `vec` flag the kernels branch on: it is on
  when every row width a kernel steps by is a multiple of `W` and off
  otherwise, so a width of 3 or 33 takes a completely scalar path and still
  matches upstream at the tested tolerances. `tests/test_ops.py` runs both
  paths at widths 1, 2, 3, 5, 7, 8, 16 and 33.
- **`gspmm` reduces a whole vector of columns at a time.** Upstream's loop
  shape is column-outer, edge-inner, which makes every one of the 32M
  (edge, column) pairs its own scalar gather. The accumulator runs across the
  columns of one destination, so a gathered row feeds all 8 vectors at once.
- **`edge_softmax` walks each node's CSR row once for all heads.** The head
  axis is the inner loop, `W` lanes at a time over the contiguous `d`-wide
  logit row, and the running max and sum live in a two-`d` scratch row the
  caller owns — so a node's row is traversed once per pass instead of once per
  head, and a head is a vector load rather than a scalar gather. One head
  still takes a separate path: with a single column the max and the sum stay
  in registers, and routing it through the scratch cost 1.7x, so `d == 1` is
  dispatched to the register version. `tests/test_ops.py` runs the multi-head
  path at 1, 3, 4 and 8 heads, for both `norm_by`, with and without an `eids`
  subset.
- **`gather_mm`'s inner product is a vector loop over the output columns.**
  The contraction is over `k` and the output is over `j`, so a `W`-wide load
  takes `W` adjacent output columns of one `b` row at once — a quarter of the
  loads the scalar form issued, over the same `d1 * d2` block, which stays in
  L1 for the whole row. The `b` block is *not* re-read per output column.
- **`gather_mm` picks its route on the size of the `b` block, not on `N`.**
  Two routes compute the same product: the fused kernel streams `a` and `dst`
  and gathers the block per row; the sorted route permutes the rows by
  `idx_b` so rows sharing a block walk it once, but pays a random-access
  gather of `a` and a random-access scatter of the result on top. At 1M rows
  and 64 blocks the fused kernel is 4.6x ahead at 16x16, 1.6x at 32x32, 1.2x
  at 64x32 and 64x64, and level at 128x128; the sorted route only takes it at
  256x256, where the block is 32 MB and no longer survives in cache. The cut
  sits at 16384 doubles. Upstream's own cut is on `N` and `D1`/`D2 > 8`, which
  sends 16x16 to the sorted route; that choice is a performance one and
  `tests/test_ops.py` pins both routes to the same numbers.
- **The op and reducer dispatch is compile-time.** `op` and `reduce_op` are
  parameters of a generic `def`, not `Int` arguments compared inside the inner
  loop, so `_gspmm` is instantiated once per (message op, reducer) pair and
  its inner loop is a bare load, combine and store.
- **`segment_softmax` is one kernel instead of five passes and two
  temporaries.** `dgl/ops/segment.py` composes it out of segment-max,
  `repeat`, subtract, `exp`, segment-sum, `repeat` and divide. The kernel does
  max, exp-and-sum, divide over the segment's own rows, with the per-column
  max and the running sum in two `d`-long scratch rows the caller owns. Same
  arithmetic, same summation order over the same rows, so the result is the
  composition's result; `tests/test_ops.py` checks it against upstream at
  widths 1, 3, 4 and 32, with zero-length segments, and with every row in one
  segment.
- **`mean` divides inside the kernel.** `gspmm` takes the in-degree array and
  divides the reduced row by it, clamped at one, instead of writing the row
  and handing a second full pass to numpy. Same IEEE division, same result.
- **`edge_softmax` skips the per-edge select when `eids` is `ALL`.** The
  Python side says so in one `Int` and the mask load disappears from all three
  passes; the mask array itself is not even allocated on that path, and a
  subset selection is what builds one.
- **Less redundant work around the kernels.** The two gathered CSR views are
  built once in `_build_csr` instead of being re-gathered from the edge list on
  every call (a 1M-element fancy index each time), and the outputs that the
  kernel writes in full are `np.empty` rather than `np.zeros`.

One kernel is scalar on purpose, and the code says why:
`replace_inf_with_zero`, because this dialect's SIMD comparison folds to a
single `Bool` rather than a lane mask and there is no `simd_select` to zero
one lane with. `gsddmm`'s `dot` accumulates a whole vector of partial products
and does one `reduce_add` per edge instead of one per vector; `W` is 4 on this
AVX2 machine and there is no AVX-512 to ask for.

### What was tried and did not pay

Each of these was built, measured with `pixi run bench`'s lock, and reverted
because it was not faster; the code is as it was found.

- **`gspmm` with the edge loop outside the column loop** (the accumulator a
  caller-owned `dout` row, so one in-edge contributes all its columns before
  the next starts). Theoretically more loads in flight, measured 1.26x-1.41x
  *slower*: the L1 round trip on the accumulator costs more than the index
  loads it saves.
- **`gspmm` unrolled 2x and 4x over columns** (two or four reduction chains,
  one index load for all of them). 2x was 0.97x-1.03x, i.e. noise; 4x spilled
  registers and was 1.21x-1.46x slower.
- **`gspmm` pairing two destinations** so two independent chains run at once.
  0.96x-1.01x. The wall is memory, not instructions: at `d = 32` the kernel
  already moves 563 MB in the time `gsddmm` moves 768 MB, and both sit at the
  single-core random-gather bandwidth of this box. More instruction-level
  parallelism does not buy more memory-level parallelism here, and
  `gsddmm` reaches its 7 GB/s only because it is edge-parallel with no
  reduction, so it can issue a whole gathered row at once.


`segment_reduce` and `segment_softmax` were left alone deliberately. Both are
one sequential scan over 256 MB of value rows with a cheap inner body, and
`segment_softmax` is additionally bound by 32M `exp` calls: a single
`np.exp` pass over the same array costs more than the kernel's three passes
put together, so there is no room for a fusion to recover.

### Parallelism and the GPU

- **Parallelism is available and used, but does not pay here.** `parallelize`
  is no longer in `std` on this toolchain; it lives in the `max` package, which
  this environment now depends on (`max = "==26.7.0.dev2026092905"`, released in
  lockstep with the `mojo` pin):

  ```mojo
  from max.algorithm import parallelize

  def work(i: Int) {imm}:      # a plain nested `def`, not a `@parameter`
      ...
  parallelize(work, workers, workers)
  ```

  Two details from `MOJO_NOTES.md` section 5b both bite: the `[func]` form no
  longer type-checks, and a `@parameter` closure is rejected as `capturing thin`.
  It works, and on a bandwidth-bound elementwise kernel it scales — a
  10M-double `dst[i] = src[i] * 2.0` measured 24.2 ms at one worker and
  14.1 ms at sixteen, 1.7x, on this box.

  It does **not** pay on these kernels, and that is a measurement, not an
  excuse. Splitting `gspmm` over destination nodes and running it at 1, 2, 4,
  8, 16, 32 and 64 workers measured 90.8, 89.7, 88.1, 92.5, 89.1, 90.0 and
  90.1 ms — flat. The reason is that `gspmm`, `gsddmm` and the segment
  operators are all memory-bound on random gathers, and the box was carrying a
  load average of 85 against 72 cores for the whole measurement, so the extra
  workers had no idle cores to run on and no spare memory bandwidth to
  contend for. Adding threads to a kernel that is already at the single-core
  random-gather bandwidth cannot help, and the evidence above says so directly.
  The parallel form is therefore left out of the shipped kernels rather than
  committed on the strength of a flag that measures flat.
- **No GPU path.** The GPU surface moved out of `std` into the same `max`
  package, so `from std.gpu import thread_idx` no longer resolves, but
  `from max.gpu import block_idx, thread_idx` and
  `from max.gpu.host import DeviceContext` do. The box does have an NVIDIA GPU.
  A GPU path is still not shipped: these kernels run at roughly 1-2 flops per
  byte, they are already at the machine's memory bandwidth, and a
  copy-in/copy-out path would move the same bytes twice. The port is CPU-only,
  and that is a bandwidth argument rather than a missing-API one.

## How it works

**FFI.** `build/build.sh` runs `mojo build --emit shared-lib -I src src/capi.mojo
-o dist/libmojo-dgl.so`. The kernels live in `src/ported.mojo` and take
`Pointer[Float64, AnyOrigin[mut=True]]`; `src/capi.mojo` is a thin `@export`
layer whose only job is rebuilding pointers from the addresses Python hands
over, because `@export` refuses a parameterised function and a pointer with an
inferred origin is parameterised. Python reaches the library through
`python/mojodgl/_lib.py`, a ctypes table of the ten exported symbols with
their `Int`-address argument lists. No symbol allocates: every buffer,
including scratch like the segment offsets, `segment_softmax`'s two `d`-long
rows and `edge_softmax`'s max-and-sum row, is a numpy array the caller owns,
so there is nothing to free and nothing to leak. `capi.mojo` is also where the
`vec` flag is decided, since it is the only layer that still has the
addresses.

**Memory layout.** Everything is C-contiguous `float64`; a feature block is `n`
rows of `d` columns and crosses as one flat address with `n` and `d` passed
separately. Node ids are `int64`; the graph keeps its edge list in the order it
was supplied (which is what upstream does) and derives the two CSR views it
needs — destination-major for message aggregation and for `norm_by="dst"`, and
source-major for `norm_by="src"`. Feature arrays are never copied on the way in
or out: `ndata`/`edata` hold the numpy array itself and pass `arr.ctypes.data`.

The one thing the layout does not give the kernels is a 32-byte-aligned base:
numpy hands out 16-byte-aligned buffers, so the vector loads are element
aligned rather than vector aligned. This toolchain emits plain unaligned
vector moves for them, they measure the same speed as aligned ones, and
`tests/test_ops.py` runs the kernels on views whose base is deliberately one
element into their allocation.

**Where the work happens.** `gspmm` is the interesting one: a single pass over
the in-edge CSR that gathers the source feature, combines it with the edge
feature and reduces on the spot, so the per-edge message never exists as an
array. `gsddmm` is the complement — edge-parallel, no reduction, and the basis
for every `u_op_v` / `v_op_e` message. `edge_softmax` is a fused three-pass
scan per node (max, exp and sum, divide) whose inner loop is the head axis, so
one pass covers every head. The segment operators take an offsets array, which
is exactly the `cumsum` of the segment lengths that `dgl/ops/segment.py` builds
in Python.
