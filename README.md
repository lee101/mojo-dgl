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
| `dgl.function` | `TargetCode`, `BuiltinFunction`, `MessageFunction`, `BinaryMessageFunction`, `CopyMessageFunction`, `copy_u`, `copy_e`, all 30 generated `u_add_v` … `e_dot_v` message builtins, `ReduceFunction`, `SimpleReduceFunction`, `sum` / `max` / `min` / `mean` |
| `dgl.ops.sddmm` | `gsddmm`, `reshape_lhs_rhs`, all 30 generated `u_add_v` … `e_dot_v` SDDMMs, `copy_u`, `copy_v`, `copy_e` |
| `dgl.ops.spmm` | `gspmm` and all 24 generated `u_{add,sub,mul,div}_e_{sum,max,min,mean}`, `copy_u_*`, `copy_e_*` |
| `dgl.ops.segment` | `segment_reduce`, `segment_softmax`, `segment_mm` |
| `dgl.ops.edge_softmax` | `edge_softmax(graph, logits, eids=ALL, norm_by="dst")` |
| `dgl.ops.gather_mm` | `gather_mm(a, b, *, idx_b)`, both the fused and the segment-`mm` path |
| `dgl.nn.functional` | `edge_softmax` (upstream 2.x re-exports it from `dgl.ops`) |

That is 64 names in `mojodgl.ops` alone. On the Mojo side `src/ported.mojo` is
the kernel file, laid out in upstream's own source order: the array primitives
first (they are what `dgl/ops/segment.py` is written against), then
`dgl/function`, then `dgl/ops` in the order `ops/__init__.py` imports it.

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
- **`edge_softmax` logits.** Upstream accepts `(E, *, *)`; this port accepts
  1-D or `(E, 1)`, which is what GAT and attention layers actually use. Upstream
  also returns a buffer as long as `logits` with the selected results in the
  first `len(eids)` positions, and so does this port — the docstring example
  that shows a 4-row result is stale relative to the shipped kernel.
- **`pull`.** Works with a builtin message function only; upstream also accepts
  a UDF. It runs the same fused kernel over the induced in-edge subgraph and
  writes back only the requested rows.
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

The table below is one `pixi run bench` invocation on one graph of 200,000 nodes
and 1,000,000 random edges, float64 features, 32 columns. Both libraries get the
same arrays, each number is the best of three runs, and nothing here is
extrapolated. Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores,
Linux x86_64, AVX2, torch 2.3.1.post100, dgl 2.3.0. `W`, the SIMD width these
kernels are built at, is 4 float64 lanes here.

| case | mojo-dgl | dgl | speedup |
| --- | ---: | ---: | ---: |
| `u_mul_e_sum` (1M edges, 32d) | 269.6 ms | 68.7 ms | 0.25x |
| `u_add_e_mean` (1M edges, 32d) | 292.9 ms | 141.4 ms | 0.48x |
| `u_div_e_min` (1M edges, 32d) | 396.6 ms | 400.9 ms | **1.01x** |
| `copy_u_max` (1M edges, 32d) | 167.0 ms | 95.4 ms | 0.57x |
| `copy_e_mean` (1M edges, 32d) | 151.7 ms | 127.0 ms | 0.84x |
| `u_dot_e` (SDDMM, 1M edges, 32d) | 158.3 ms | 51.0 ms | 0.32x |
| `u_mul_v` (SDDMM, 1M edges, 32d) | 295.7 ms | 108.8 ms | 0.37x |
| `edge_softmax` norm_by=dst (1M edges) | 49.9 ms | 39.9 ms | 0.80x |
| `edge_softmax` norm_by=src (1M edges) | 48.8 ms | 34.9 ms | 0.71x |
| `segment_reduce` sum (1M rows, 32d, 200k segments) | 85.9 ms | 60.3 ms | 0.70x |
| `segment_reduce` max (1M rows, 32d, 200k segments) | 158.4 ms | 177.5 ms | **1.12x** |
| `segment_softmax` (1M rows, 32d, 200k segments) | 642.5 ms | 808.1 ms | **1.26x** |
| `gather_mm` (1M rows, 16x16, 64 blocks) | 1010.6 ms | 3763.6 ms | **3.72x** |
| `update_all` `u_mul_e` + `sum` (1M edges, 32d) | 314.1 ms | 93.8 ms | 0.30x |
| `update_all` `copy_u` + `max` (1M edges, 32d) | 218.9 ms | 167.3 ms | 0.76x |

Read those ratios with the load in mind. This port is single-threaded; dgl's
kernels are OpenMP-parallel and spread over 72 cores, so the two-gather cases
are lost on memory latency rather than on arithmetic: `u_mul_e_sum` and
`u_mul_v` each do two random row reads per edge, and dgl gets 72 cores to hide
that latency behind. The cases this port wins are the ones with a
single sequential scan, where there is no latency to hide and dgl's threads
only add bookkeeping. Which cases those are moves with the box's load: the same
binary measured `gather_mm` at 1.24x and `segment_reduce` max at 1.95x on
earlier runs the same day, and at 3.72x and 1.12x on the run above. The box is
shared, and re-running moves every number in the table in both columns, so treat a single
ratio as an order of magnitude, not a measurement of the code.

### Design decisions that follow from that

- **The column loop is a vector loop.** `gspmm`, `gsddmm`, `segment_reduce`,
  `segment_softmax` and `segment_mm` walk the feature columns `W` lanes at a
  time (`p.unsafe_load[width=W]`, `p.unsafe_store`, free `min`/`max`, which
  work on `SIMD`), with a scalar tail for the `d % W` columns that do not fill
  a vector. `capi.mojo` sets the `vec` flag the kernels branch on: it is on
  when every row width is a multiple of `W` and off otherwise, so a width of
  3 or 33 takes a completely scalar path and still matches upstream at the
  tested tolerances. `tests/test_ops.py` runs both paths at widths 1, 2, 3, 5,
  7, 8, 16 and 33.
- **`gspmm` reduces a whole vector of columns at a time.** Upstream's loop
  shape is column-outer, edge-inner, which makes every one of the 32M
  (edge, column) pairs its own scalar gather. The accumulator runs across the
  columns of one destination, so a gathered row feeds all 8 vectors at once.
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
  passes; the mask array itself is still what a subset selection uses.
- **Less redundant work around the kernels.** The two gathered CSR views are
  built once in `_build_csr` instead of being re-gathered from the edge list on
  every call (a 1M-element fancy index each time), and the outputs that the
  kernel writes in full are `np.empty` rather than `np.zeros`.

Two things were tried and dropped because they did not measure faster:
unrolling the `gspmm` gather loop to get more cache misses in flight, and a
horizontal reduction per vector inside the dot product (`reduce_add` is a
handful of shuffles, so at the contraction widths used here it lost to a
vector accumulator reduced once per edge). Two more stayed scalar on purpose:
`replace_inf_with_zero`, because this dialect's SIMD comparison folds to a
single `Bool` rather than a lane mask and there is no `simd_select` to zero one
lane with, and `gather_mm`, which is a small-matrix gather that vectorises
badly. `W` is 4 on this AVX2 machine; there is no AVX-512 to ask for.

### Parallelism and the GPU: neither, on purpose

- **No parallelism.** `parallelize` is gone from this toolchain
  (`from std.algorithm import parallelize` fails with "package 'algorithm'
  does not contain 'parallelize'"), there is no `std.parallelism` or
  `std.threading`, and there is no `max` package to import at all
  (`import max` -> "unable to locate module 'max'"). There is no replacement
  to put behind a size threshold, so the work stays serial.
- **No GPU path.** This toolchain has no `std.gpu` and no `std.gpu.host` at
  all -- `from std.gpu import thread_idx` fails to resolve, let alone
  `DeviceContext`, `enqueue_create_buffer` or `synchronize`. The box does have
  an NVIDIA GeForce RTX 5090, so this is not a memory decision: the host API
  to launch a kernel does not exist here, and a GPU path that cannot be built
  and run is not worth shipping. The port is CPU-only.

## How it works

**FFI.** `build/build.sh` runs `mojo build --emit shared-lib -I src src/capi.mojo
-o dist/libmojo-dgl.so`. The kernels live in `src/ported.mojo` and take
`Pointer[Float64, AnyOrigin[mut=True]]`; `src/capi.mojo` is a thin `@export`
layer whose only job is rebuilding pointers from the addresses Python hands
over, because `@export` refuses a parameterised function and a pointer with an
inferred origin is parameterised. Python reaches the library through
`python/mojodgl/_lib.py`, a ctypes table of the ten exported symbols with
their `Int`-address argument lists. No symbol allocates: every buffer, including
scratch like the segment offsets and the two `d`-long rows `segment_softmax`
reduces into, is a numpy array the caller owns, so there is nothing to free and
nothing to leak. `capi.mojo` is also where the `vec` flag is decided, since it
is the only layer that still has the addresses.

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
for every `u_op_v` / `v_op_e` message. `edge_softmax` is a fused three-pass scan
per node (max, exp and sum, divide) over the same CSR. The segment operators
take an offsets array, which is exactly the `cumsum` of the segment lengths that
`dgl/ops/segment.py` builds in Python.
