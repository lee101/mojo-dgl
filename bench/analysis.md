# mojo-dgl — ANALYZE phase: where the time goes

Measured map only. No algorithm was changed in this phase (nothing in `src/` or
`tests/` was touched). Every number below comes from a run in this session.

## Machine and method

- Machine (from `bench/bench.py`): Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 cores
  (2 sockets × 18 cores × 2 HT), Linux x86_64, AVX2 (so `W = 4` float64 lanes),
  torch 2.3.1.post100, dgl 2.3.0, numpy 1.26.4, mojo pin in `pixi.toml`.
- Shapes are exactly `bench/bench.py`: NODES=200 000, EDGES=1 000 000, DIM=32,
  float64, CSR built once by `DGLGraph._build_csr` (not timed).
- `pixi run bench` behaviour: `pixi run bench` itself only prints the
  "run through `pixi run bench`" guard in this checkout's task wiring
  (`bench = flock ... python bench/bench.py`, `bench-here = ... --locked`), so
  the flock-held invocation used was `pixi run bench-here` — same
  `/tmp/mojo-bench.lock`, same `bench/bench.py --locked`. Two full runs were
  recorded (A and B below); they differ because the box is shared and upstream
  dgl runs multithreaded (OpenMP over the box) while this port's Mojo kernels
  split over `WORKERS=32` (see "Parallelism" below).
- Deeper timing (`bench/profile_deep.py`, run under the same flock, best-of-5
  after warmup, same arrays/seed as `bench.py`): each public entry was timed
  against (1) upstream dgl on identical arrays, (2) a plain numpy baseline on
  the same arrays, and (3) a "kernel-only" timing — the same Mojo kernel called
  directly through ctypes with a pre-allocated output, bypassing all Python
  glue. `public − kernel-only` is the glue estimate; where the two were measured
  in different batches the interleaved A/B check in `/tmp/glue_check.py` is
  cited instead. Bytes/flops below are computed from the measured shapes;
  "logical bytes" = the bytes the kernel requests (counting each gather, cache
  hits included); "physical bytes" for `gather_mm` counts the `b` block once
  because it stays in L1 for the 16×16 shape.
- Roofline reference for this box: single-core AVX2 float64 peak ≈ 2.3 GHz ×
  8 lanes × 2 FMA = 36.8 GFLOP/s; a single core streams ≈ 15–20 GB/s. A kernel
  sitting at ~10–20 logical GB/s with ≤0.25 flop/byte is single-core
  bandwidth-bound, not compute-bound.

## What was run

1. `pixi run bench-here` (flock `/tmp/mojo-bench.lock`), twice — table in
   "Bench runs" below.
2. `bench/profile_deep.py` under the same flock — public vs upstream vs numpy
   vs kernel-only for 22 cases, plus ctypes-call counts, per-call numpy
   allocation bytes, and small-size scaling (1k/10k/100k edges).
3. Glue isolation at large size: `/tmp/glue_check.py` (gspmm, gsddmm,
   segment_reduce interleaved public vs kernel-only), `/tmp/gluework/glue2.py`
   (edge_softmax, segment_softmax, segment_reduce sum/mean/max, gather_mm),
   `/tmp/gluework/glue5.py` + cProfile (segment_reduce mean/max breakdown),
   and `/tmp/glue3.py` (offsets/clamp/replace-inf micro-costs, `eids` subset).

## Ranked hotspot table

Ranked by measured mojo-dgl public-call time at the bench shape (run B /
deep-run best-of-5; the two agree within run-to-run noise except where noted).
"Slower than upstream" = mojo best > upstream best in **both** full bench runs
(solid) or in one (borderline, marked).

| # | function (public entry) | input size | time (mojo, public) | bytes touched (logical) | flops | flops/byte | bound | time in Python/ctypes glue or in kernel |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `segment_softmax` | 1M rows × 32, 200k segments | 62.1 ms (deep) / 63.0 ms (B) / 45.7 ms (A) | 770 MB (read 256 MB + write 256 MB + 3rd pass read 256 MB) | 96M (max + exp + div per element) | 0.12 | memory-bound (exp adds scalar cost) | kernel 29–40 ms; rest is glue/allocation (below) |
| 2 | `copy_u` SDDMM (`ops.copy_u`, gsddmm copy_lhs) | 1M edges × 32 | 68.4 ms (deep) | 448 MB (gather 256 + write 256, binary gather of dst rows skipped on copy path still reads uids/vids) | 0 (copy) | 0.0 | memory-bound | kernel-dominated |
| 3 | `u_mul_v` SDDMM | 1M edges × 32 | 55.6 ms (deep) / 38.1 ms (B) / 51.6 ms (A) | 784 MB (2×256 gather + 256 write + 16 idx) | 32M | 0.04 | memory-bound | kernel 21.1 ms (separate batch); interleaved glue ≈ 0 (public 33.7 vs kernel 37.3 — noise dominates; glue < ~2 ms) |
| 4 | `u_add_v` SDDMM (extra case, same kernel) | 1M edges × 32 | 54.4 ms (deep) | 784 MB | 32M | 0.04 | memory-bound | kernel-dominated |
| 5 | `gspmm` family: `u_mul_e_sum` representative | 200k dst × 32, 1M edges | 34.6 ms (deep) / 32.7 ms (B) / 30.4 ms (A) | 581 MB (2×256 gather + 51 dst + 16 idx/eid + 1.6 indptr) | 64M (mul+add per edge-elem) | 0.11 | memory-bound | kernel 26.6 ms; glue 1.9 ms (interleaved 36.4 vs 34.5) |
| 6 | `u_add_e_mean` (gspmm, mean folded) | same | 35.4 ms (deep) / 36.8 ms (B) | 581 MB | 64M | 0.11 | memory-bound | kernel-dominated; mean divide is inside kernel (no 2nd numpy pass) |
| 7 | `u_div_e_min`, `copy_e_mean`, `u_sub_e_sum`, `u_mul_e_max` (same gspmm kernel, other op/reduce) | same | 36.0 / 27.3 / 35.9 / 35.1 ms (deep); bench B 29.3 / 25.4 / — / — | 581 MB (copy_e: 325 MB — no lhs gather) | 64M (32M for copy) | 0.11 (0.10) | memory-bound | kernel-dominated (1 ctypes call each) |
| 8 | `copy_u_max` (gspmm copy_lhs) | same | 20.3 ms (deep) / 15.9 ms (B) / 21.7 ms (A) — fastest gspmm | 325 MB | 0 (compare only) | 0.10 | memory-bound | kernel-dominated (1 ctypes call) |
| 9 | `segment_reduce` sum | 1M × 32, 200k seg | 26.5 ms (deep) / 19.2 ms (B) / 25.4 ms (A) | 309 MB (256 read + 51 write + 1.6 offsets) | 32M | 0.10 | memory-bound | kernel 13.4–15.7 ms; glue 8–12 ms (table below) |
| 10 | `segment_reduce` max / min | same | 37.9 / 37.6 ms (deep); bench B max 33.7 ms | 309 MB + 51 MB `replace_inf` pass (max/min only) | 32M | 0.09 | memory-bound | kernel + `replace_inf_with_zero` ctypes pass (~17.5 ms incl. `np.zeros`) |
| 11 | `segment_reduce` mean | same | 45.3 ms (glue5) / ~40.5 ms (glue3) | 309 MB + 51 MB numpy divide | 32M + 6.4M div | 0.10 | memory-bound | kernel ~15 ms + numpy in-place divide 13.1 ms |
| 12 | `gather_mm` (fused, 16×16, 64 blocks) | 1M rows, 16×16 | 22.6 ms (deep) / 22.7 ms (B) / 30.2 ms (A) | 264 MB physical (128 a + 128 dst + 8 idx; `b` is 131 KB, L1-resident) / 2 312 MB logical | 512M (2·16·16 per row) | 1.94 physical / 0.22 logical | memory-bound (physical); the only case near compute relevance | kernel 14.6–18.7 ms; remainder is f64/i64 checks + min/max |
| 13 | `u_dot_e` SDDMM (dot) | 1M edges × 32 | 26.7 ms (deep) / 23.1 ms (B) / 24.7 ms (A) | 536 MB (2×256 gather + 8 write + 16 idx) | 64M (mul+add, 1 reduce per edge) | 0.12 | memory-bound | kernel 19.6–22.2 ms; glue ≈ 4 ms |
| 14 | `edge_softmax` 8 heads | 1M × (8,1) | 33.0 ms (deep) / 32.1 ms (B) / 30.4 ms (A) | 336 MB (3 passes over 64 MB) | 24M | 0.07 | memory-bound | kernel-dominated (multi-head scratch path) |
| 15 | `edge_softmax` norm_by=dst, d=1 | 1M × 1 | 5.9 ms (deep) / 12.7 ms (B) / 10.2 ms (A) — **smallest absolute time, worst ratio** | 58 MB (3 passes over 8 MB + 16 idx) | 3M | 0.05 | memory-bound | kernel 7.3–7.9 ms ≈ public (glue negligible: f64 0.002 ms, dst alloc 0.003 ms) |
| 16 | `edge_softmax` norm_by=src, d=1 | 1M × 1 | 10.1 ms (deep) / 10.7 ms (B) / 11.7 ms (A) | 58 MB | 3M | 0.05 | memory-bound | kernel-dominated |
| 17 | `update_all` u_mul_e + sum (fused gspmm) | 1M edges × 32 | 29.9 ms (deep) / 26.0 ms (B) / 30.8 ms (A) | 581 MB | 64M | 0.11 | memory-bound | 1 ctypes call; fused path, glue ≈ 0 |
| 18 | `update_all` copy_u + max (fused gspmm) | 1M edges × 32 | 32.7 ms (deep) / 31.8 ms (B) / 35.0 ms (A) | 325 MB | 0 | 0.10 | memory-bound | 2 allocs (102 MB); fused, glue small |
| 19 | `dgl.graph` construction (not in `bench.py`) | 1M edges, 200k nodes | 305 ms (deep: 284–306 ms across batches) | ~50 MB (argsort ×2, bincount ×2, gathers) | — (sort-dominated) | — | compute-bound (numpy argsort) | pure Python/numpy in `heterograph._build_csr`; no Mojo kernel |
| 20 | `edge_softmax` with `eids` subset (half the edges) — NOT in `bench.py` | 500k of 1M edges | 710 ms (glue3: 709.9 ms) | 58 MB + Python dict grouping | 3M | 0.05 | **Python-loop-bound** | **pure Python** (`edge_softmax.py:137–145` loop over `order.tolist()`), kernel is ~8 ms of it |

### Kernel-only vs public glue (large size, interleaved where it matters)

| case | public (best) | kernel-only (best, direct ctypes) | glue estimate | glue breakdown (measured) |
| --- | ---: | ---: | ---: | --- |
| gspmm `u_mul_e_sum` | 36.4 ms | 34.5 ms | ~1.9 ms | `f64` 0.002 ms, dst `np.empty` 51 MB 0.012 ms, CSR fetch ~0 ms, 1 ctypes call; raw ctypes overhead 5.6 µs (tiny `mdgl_cumsum`) |
| gsddmm `u_mul_v` | 33.7 ms | 37.3 ms | ≈ 0 (noise) | same shape as gspmm; 1 ctypes call |
| segment_reduce sum | 24.1 ms | 15.7 ms | 8.4 ms | `_offsets` 0.39 ms (concat + `mdgl_cumsum` 200k = 0.60 ms) + `_check_offsets` 0.23–0.88 ms (`np.diff` over 200k) + `_flat`/`f64`/`i64` ~0.005 ms + dst alloc; the rest is batch noise. ctypes calls = 2 (not 1) |
| edge_softmax dst d=1 | 7.8–9.6 ms | 7.3–8.3 ms | ~0.5–1 ms | `f64` 0.002 ms, `mdgl_workers` 0.4 µs, scratch alloc — all negligible; 2 ctypes calls (workers + kernel) |
| segment_softmax | 56–62 ms | 29–40 ms | 16–33 ms (batch-dependent) | dst is `np.zeros` 256 MB (alloc 0.011 ms, page-fault on 1st write is real but amortised), `mdgl_workers` + scratch, `_offsets`/`_check_offsets` ~1 ms; remainder is run-to-run spread on the shared box — flagged as uncertain, not as a precise glue figure |
| gather_mm | 22.6 ms | 14.6–18.7 ms | 4–8 ms | `f64`/`i64` + `idx.min()/max()` over 1M ints = 0.60 ms; 1 ctypes call |
| update_all (both) | 29.9 / 32.7 ms | = gspmm kernel | ≈ 0 | fused single gspmm; `_pass` dispatch is pure Python, unmeasurably small vs 30 ms |

Conclusion on glue: at the bench size the Python/ctypes layer is **not** the
hotspot. Raw ctypes cost is 5.6 µs/call; even the heaviest glue path
(segment_reduce with offsets + checks) is ~1–2 ms of a 24 ms call, i.e.
< 10%. The segment_reduce **mean/max** variants are the exception and their
extra cost is *second-pass numpy/ctypes work*, not FFI overhead: mean adds a
13.1 ms in-place numpy divide over 51 MB, max/min add a
`replace_inf_with_zero` pass (17.5 ms incl. its `np.zeros` over 51 MB;
48 ms when the input was an all-inf array — that number is the same kernel on
degenerate data and is not comparable). Both are counted in rows 10–11 above.

## Bench runs (both retained; rows not dropped)

Run A (first successful `pixi run bench-here`):

| case | mojo-dgl | dgl | speedup |
| --- | ---: | ---: | ---: |
| u_mul_e_sum (1M edges, 32d) | 30.4 ms | 28.5 ms | 0.94x — **slower** |
| u_add_e_mean (1M edges, 32d) | 26.4 ms | 40.1 ms | 1.52x |
| u_div_e_min (1M edges, 32d) | 32.1 ms | 176.7 ms | 5.51x |
| copy_u_max (1M edges, 32d) | 21.7 ms | 45.3 ms | 2.09x |
| copy_e_mean (1M edges, 32d) | 24.9 ms | 35.2 ms | 1.42x |
| u_dot_e (SDDMM, 1M edges, 32d) | 24.7 ms | 12.8 ms | 0.52x — **slower** |
| u_mul_v (SDDMM, 1M edges, 32d) | 51.6 ms | 49.5 ms | 0.96x — **slower** |
| edge_softmax norm_by=dst (1M edges) | 10.2 ms | 3.4 ms | 0.33x — **slower** |
| edge_softmax norm_by=src (1M edges) | 11.7 ms | 2.7 ms | 0.23x — **slower** |
| edge_softmax 8 heads (1M edges, (E, 8, 1)) | 30.4 ms | 23.0 ms | 0.76x — **slower** |
| segment_reduce sum (1M rows, 32d, 200k segments) | 25.4 ms | 18.6 ms | 0.73x — **slower** |
| segment_reduce max (1M rows, 32d, 200k segments) | 39.8 ms | 62.6 ms | 1.57x |
| segment_softmax (1M rows, 32d, 200k segments) | 45.7 ms | 360.8 ms | 7.90x |
| gather_mm (1M rows, 16x16, 64 blocks) | 30.2 ms | 179.6 ms | 5.95x |
| update_all u_mul_e + sum (1M edges, 32d) | 30.8 ms | 25.3 ms | 0.82x — **slower** |
| update_all copy_u + max (1M edges, 32d) | 35.0 ms | 72.6 ms | 2.07x |

Run B (latest `pixi run bench-here`, same lock and harness):

| case | mojo-dgl | dgl | speedup |
| --- | ---: | ---: | ---: |
| u_mul_e_sum (1M edges, 32d) | 32.7 ms | 49.0 ms | 1.50x |
| u_add_e_mean (1M edges, 32d) | 36.8 ms | 45.1 ms | 1.23x |
| u_div_e_min (1M edges, 32d) | 29.3 ms | 176.2 ms | 6.01x |
| copy_u_max (1M edges, 32d) | 15.9 ms | 41.4 ms | 2.60x |
| copy_e_mean (1M edges, 32d) | 25.4 ms | 30.6 ms | 1.21x |
| u_dot_e (SDDMM, 1M edges, 32d) | 23.1 ms | 11.8 ms | 0.51x — **slower** |
| u_mul_v (SDDMM, 1M edges, 32d) | 38.1 ms | 41.0 ms | 1.08x |
| edge_softmax norm_by=dst (1M edges) | 12.7 ms | 2.8 ms | 0.22x — **slower** |
| edge_softmax norm_by=src (1M edges) | 10.7 ms | 2.2 ms | 0.21x — **slower** |
| edge_softmax 8 heads (1M edges, (E, 8, 1)) | 32.1 ms | 19.4 ms | 0.61x — **slower** |
| segment_reduce sum (1M rows, 32d, 200k segments) | 19.2 ms | 16.5 ms | 0.86x — **slower** |
| segment_reduce max (1M rows, 32d, 200k segments) | 33.7 ms | 75.4 ms | 2.24x |
| segment_softmax (1M rows, 32d, 200k segments) | 63.0 ms | 293.7 ms | 4.66x |
| gather_mm (1M rows, 16x16, 64 blocks) | 22.7 ms | 176.4 ms | 7.77x |
| update_all u_mul_e + sum (1M edges, 32d) | 26.0 ms | 23.7 ms | 0.91x — **slower** |
| update_all copy_u + max (1M edges, 32d) | 31.8 ms | 52.9 ms | 1.66x |

Consistently slower than upstream (both runs): `u_dot_e` (0.51–0.52x),
`edge_softmax` dst (0.22–0.33x), src (0.21–0.23x), 8 heads (0.60–0.76x),
`segment_reduce` sum (0.73–0.86x), `update_all` u_mul_e+sum (0.82–0.91x).
Borderline (slower in A only): `u_mul_e_sum` (0.94x vs 1.50x) and `u_mul_v`
(0.96x vs 1.08x) — the box is shared and upstream is multithreaded, so the
*mojo* column is the stable one and the dgl column moves (28.5→49.0 ms for
`u_mul_e_sum` between runs); these two are ~par, not clear losses.

## Errors, other files

- `pixi run bench` (without `-here`) prints only the guard
  `run this through pixi run bench, which takes the machine-wide lock...` and
  exits without running the harness, because `pixi.toml`'s `bench` task runs
  `flock /tmp/mojo-bench.lock python bench/bench.py` without `--locked`,
  and `bench.py` raises `SystemExit` unless `--locked` is passed.
  `pixi run bench-here` (same flock + `--locked`) is the flock-wrapped path
  that executes the harness; it was used for both recorded runs under the
  same `/tmp/mojo-bench.lock`, so no benchmark ran alongside another.
- `bench/profile_deep.py` needed fixes before it completed: a `None` bug in
  the numpy mean baseline, `x+4.0` temporaries inflating timed lambdas,
  `mojodgl.ops.*` star imports shadowing the `lib` name, and a small-graph
  row-count check. It then exited 0.
- Discarded glue scripts: `/tmp/glue2.py` first version segfaulted (it passed
  a fresh `np.empty` inside the timed ctypes call); `/tmp/glue3.py`'s final
  `update_all` check used a graph without `ndata["h"]` (KeyError). Neither
  contributed numbers. `/tmp/profile.py` (a factory-owned shadow, not this
  repo's) broke `cProfile` imports by shadowing stdlib `profile.py`;
  profiling was re-run from `/tmp/gluework/` instead.
- Only `bench/analysis.md` and `bench/profile_deep.py` (the harness that
  produced its numbers) are new in the tree; nothing in `src/` or `tests/`
  was modified (`git status` clean apart from `bench/`).

## Numpy baselines (same arrays, deep run)

| case | mojo (public, best) | numpy (best) | notes |
| --- | ---: | ---: | --- |
| `u_mul_e_sum` | 34.6 ms | 1165.9 ms | `np.add.at` over 1M edges |
| `u_add_e_mean` | 35.4 ms | 1104.4 ms | add.at + divide |
| `copy_u_max` | 20.3 ms | 1634.6 ms | `np.maximum.at` |
| `u_dot_e` | 26.7 ms | 277.1 ms | `np.einsum` |
| `u_mul_v` / `u_add_v` | 55.6 / 54.4 ms | 518.5 / 533.0 ms | fancy-index gathers |
| `copy_u` SDDMM | 68.4 ms | 431.8 ms | gather |
| `edge_softmax` dst (np.at version) | 5.9 ms | 31.2 ms (separate row) | `maximum.at` + `add.at` + exp |
| `segment_reduce` sum / max | 26.5 / 37.9 ms | 601.9 / 830.9 ms | `reduceat` |
| `gather_mm` | 22.6 ms | 1062.4 ms | `einsum` over gathered blocks |

The Mojo kernels beat every numpy baseline by 10-50x; numpy is not a
competitor here and is not the bottleneck. It is retained as an algorithmic
sanity floor, not as a target.

## Reproduce

- `pixi run bench-here` (flock `/tmp/mojo-bench.lock`) -> the two tables in
  "Bench runs" (re-runs will differ; the mojo column is the stable one).
- `flock /tmp/mojo-bench.lock pixi run python bench/profile_deep.py`
  -> public vs upstream vs numpy vs kernel-only, ctypes counts, allocation
  bytes; raw JSON at `/tmp/profile_deep.json`.
