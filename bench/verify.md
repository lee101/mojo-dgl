# VERIFY phase — what was reproduced, what was not, what changed

Date: 2026-11-09. Machine for every number below:
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores, Linux x86_64, AVX2
(`W = 4`), torch 2.3.1.post100, dgl 2.3.0 (upstream 2.3.0 via pixi).
Box was shared throughout (load averages observed 36–71 across this session's
bench runs; `pixi run bench` literal invocation load ~47).

This file replaces the previous verify.md (committed `d75376e`), whose
numbers were measured in a prior session on the same box. The numbers in
this file are measured in this session only; nothing is carried over.

## Commands run (in order)

1. `pixi run build && pixi run test` — green. 295 passed in 9.30s.
   Log: `/tmp/verify-buildtest.log` (this session).
2. `pixi run bench` — **does not run the harness.** It prints only:
   `run this through pixi run bench, which takes the machine-wide lock…`
   and exits without timing any case. Same wiring bug as before:
   `pixi.toml` `bench = flock /tmp/mojo-bench.lock python bench/bench.py`
   holds the flock but never passes `--locked`, and `bench/bench.py` raises
   unless `--locked` is in `sys.argv`. The flock-held invocation that
   actually executes the harness is `pixi run bench-here`
   (`flock /tmp/mojo-bench.lock python bench/bench.py --locked`; same lock,
   same harness, best-of-3 per case). All bench numbers below use that
   invocation. Not fixed: the task says to change code only to fix a
   failing-test bug, and build/test are unaffected.
3. `pixi run bench-here` x 4 (four separate invocations, each best-of-3 per
   case). Logs: `/tmp/verify2-bench-{1..4}.log` (this session).
   Medians/ranges below are computed from those four runs only.
4. `bench/_verify_parity.py` (pre-existing helper, run as-is, unmodified):
   runs all 16 bench cases at the bench shape (200k nodes / 1M edges / 32d,
   same seeds as `bench/bench.py`) through both libraries on identical
   arrays and compares outputs. Result: **all 16 parity-OK** (see
   "Same input / no skipped work"). Log: `/tmp/verify2-parity.log`.

## README table vs measured — 5 rows outside 20%

The README table at the start of this session (28.2 ms mojo for
`u_mul_e_sum`, 40.4 ms for `u_mul_v`, 56.1 ms for `segment_softmax`, …) was
checked row by row against the median of this session's four invocations.
11 of 16 rows reproduce within 20% on both columns; 5 rows fail on at least
one column by more than 20% (see "Old README -> corrected" below). Per the
task, the whole table is replaced with this session's measured medians so
that every number in the README is one this session measured. No row was
removed or reworded to look better; slower-than-upstream rows stay slower
(see below).

### Corrected table (median of the four invocations in this session)

Each invocation is best-of-3 per case (bench.py `timeit`, `repeat=3`);
the table is the median across the four invocations.

| case | mojo-dgl | dgl | speedup | per-run mojo range | per-run dgl range |
| --- | ---: | ---: | ---: | --- | --- |
| `u_mul_e_sum` (1M edges, 32d) | 30.5 ms | 30.2 ms | 1.01x | 24.6–36.9 | 28.6–31.9 |
| `u_add_e_mean` (1M edges, 32d) | 39.6 ms | 39.8 ms | 1.02x | 38.4–40.4 | 36.9–46.1 |
| `u_div_e_min` (1M edges, 32d) | 36.8 ms | 182.2 ms | 5.29x | 28.3–42.7 | 154.5–191.1 |
| `copy_u_max` (1M edges, 32d) | 22.6 ms | 40.3 ms | 1.89x | 16.7–22.8 | 39.4–45.1 |
| `copy_e_mean` (1M edges, 32d) | 26.2 ms | 35.6 ms | 1.29x | 25.8–29.7 | 29.8–37.8 |
| `u_dot_e` (SDDMM, 1M edges, 32d) | 22.1 ms | 10.7 ms | 0.49x | 20.3–25.6 | 10.1–15.7 |
| `u_mul_v` (SDDMM, 1M edges, 32d) | 45.0 ms | 50.2 ms | 1.14x | 41.2–58.6 | 46.4–59.7 |
| `edge_softmax` norm_by=dst (1M edges) | 11.9 ms | 2.4 ms | 0.22x | 10.4–16.4 | 2.2–4.0 |
| `edge_softmax` norm_by=src (1M edges) | 12.2 ms | 2.7 ms | 0.23x | 9.5–16.0 | 2.3–3.5 |
| `edge_softmax` 8 heads (1M edges, `(E, 8, 1)`) | 29.9 ms | 22.9 ms | 0.77x | 29.3–34.0 | 19.8–23.8 |
| `segment_reduce` sum (1M rows, 32d, 200k segments) | 18.8 ms | 21.0 ms | 1.07x | 17.4–24.8 | 16.6–25.6 |
| `segment_reduce` max (1M rows, 32d, 200k segments) | 36.7 ms | 72.9 ms | 1.96x | 36.0–41.8 | 63.2–82.8 |
| `segment_softmax` (1M rows, 32d, 200k segments) | 65.8 ms | 399.6 ms | 6.27x | 59.8–75.4 | 377.0–428.6 |
| `gather_mm` (1M rows, 16x16, 64 blocks) | 28.0 ms | 169.6 ms | 7.09x | 22.5–36.6 | 153.5–236.1 |
| `update_all` `u_mul_e` + `sum` (1M edges, 32d) | 29.4 ms | 27.5 ms | 0.97x | 28.2–37.7 | 27.2–40.6 |
| `update_all` `copy_u` + `max` (1M edges, 32d) | 36.8 ms | 71.8 ms | 2.17x | 30.2–46.8 | 64.7–89.2 |

Rows where the port is slower than upstream stay slower in the table,
exactly as measured: `u_dot_e` (0.49x), `edge_softmax` dst (0.22x), src
(0.23x), 8 heads (0.77x), `update_all` `u_mul_e` + `sum` (0.97x) — five
rows. `u_mul_e_sum` (1.01x, 0.85x–1.17x across the four invocations) and
`update_all` `u_mul_e` + `sum` (0.72x–1.34x) flip sign across invocations,
so both are parity within noise on this shared box, not clear wins.
`u_add_e_mean` (0.91x–1.17x) and `segment_reduce` sum (0.95x–1.13x) are
likewise at parity. The four consistent losses (`u_dot_e`, `edge_softmax`
x3) were slower than upstream in every one of the four invocations.

Old README -> corrected (mojo / dgl / speedup), for the record:

| case | old | corrected | >20% off? |
| --- | --- | --- | --- |
| `u_mul_e_sum` | 28.2 / 27.4 / 0.97x | 30.5 / 30.2 / 1.01x | no (1.08x / 1.10x) |
| `u_add_e_mean` | 32.4 / 37.5 / 1.16x | 39.6 / 39.8 / 1.02x | **yes, mojo 1.22x** |
| `u_div_e_min` | 28.3 / 149.8 / 5.30x | 36.8 / 182.2 / 5.29x | **yes, 1.30x / 1.22x** |
| `copy_u_max` | 19.0 / 39.1 / 2.07x | 22.6 / 40.3 / 1.89x | no (1.19x / 1.03x) |
| `copy_e_mean` | 26.2 / 30.9 / 1.18x | 26.2 / 35.6 / 1.29x | no (1.00x / 1.15x) |
| `u_dot_e` | 21.8 / 10.6 / 0.49x | 22.1 / 10.7 / 0.49x | no (1.01x / 1.01x) |
| `u_mul_v` | 40.4 / 47.2 / 1.17x | 45.0 / 50.2 / 1.14x | no (1.11x / 1.06x) |
| `edge_softmax` dst | 10.8 / 2.4 / 0.22x | 11.9 / 2.4 / 0.22x | no (1.11x / 0.98x) |
| `edge_softmax` src | 10.9 / 2.3 / 0.22x | 12.2 / 2.7 / 0.23x | no (1.12x / 1.17x) |
| `edge_softmax` 8 heads | 32.4 / 19.4 / 0.60x | 29.9 / 22.9 / 0.77x | no (0.92x / 1.18x) |
| `segment_reduce` sum | 19.2 / 19.2 / 1.00x | 18.8 / 21.0 / 1.07x | no (0.98x / 1.09x) |
| `segment_reduce` max | 39.2 / 60.0 / 1.53x | 36.7 / 72.9 / 1.96x | **yes, dgl 1.22x** |
| `segment_softmax` | 56.1 / 286.2 / 5.10x | 65.8 / 399.6 / 6.27x | **yes, dgl 1.40x** |
| `gather_mm` | 23.7 / 141.8 / 5.98x | 28.0 / 169.6 / 7.09x | no (1.18x / 1.20x) |
| `update_all` `u_mul_e` + `sum` | 33.5 / 24.0 / 0.72x | 29.4 / 27.5 / 0.97x | no (0.88x / 1.15x) |
| `update_all` `copy_u` + `max` | 36.9 / 54.1 / 1.47x | 36.8 / 71.8 / 2.17x | **yes, dgl 1.33x** |

Note `gather_mm`'s dgl column is 1.20x the old figure — on the 20% boundary
(ratios computed from the printed medians; 169.6/141.8 = 1.196). It is
replaced anyway under the whole-table rule above.

## Same input / no skipped work — checked

- **Same input, both sides.** `bench/bench.py:data()` builds one graph and
  its feature arrays (`u`, `v`, `x`, `e`, `z`, `zh`, all from
  `default_rng(0)`) and the pair `(g, ug)` — mojodgl gets them as numpy,
  upstream as `torch.as_tensor` of the *same* buffers. `segment_data()`
  (seed 11) and `mm_data()` (seed 12) are likewise shared. Every `CASES`
  lambda passes the identical arrays to both sides (verified by reading all
  16 `CASES` defs: `t(a) = torch.as_tensor(a)`, zero-copy view). The
  `u_div_e_min` case shifts both `x` and `e` by `+4.0` before either side
  sees them — that shift is applied once, outside the timed lambda, to the
  shared arrays.
- **Outputs match at the bench shape.** `bench/_verify_parity.py` ran all 16
  cases at 200k nodes / 1M edges / 32d through both libraries and compared:
  all 16 `allclose(rtol=1e-9, atol=1e-9)` = True, max abs diff <= 5.4e-13
  wherever both sides are finite. Two cases (`u_div_e_min`, `copy_u_max`)
  report `maxabsdiff=nan` in the script because entries are `+inf` on
  *both* sides (zero-in-degree rows — upstream's `min`/`max` of an empty
  set) and `inf - inf = nan` in the diff; `allclose` is True and inf
  positions coincide on both sides. No case silently computes less on one
  side: shapes match exactly (`(200000,32)` for gspmm cases, `(1000000,1)`
  for `u_dot_e`, `(1000000,8,1)` for 8-head softmax, etc.), and the
  update_all cases run the full message+reduce on both sides.
- **No caching that skips work.** `data()`/`segment_data()`/`mm_data()` use a
  module-level `_CACHE` so the *inputs* are built once — that is input
  sharing, not result caching; each timed rep calls the public op fresh.
  `update_all` outputs are recomputed each rep.
- **One asymmetry worth recording (not a skip, not corrected).** `main()`
  warms mojodgl once (`ours()`) before timing but never warms upstream
  (`theirs()`). Best-of-3 mostly absorbs this, but the first upstream rep
  may carry lazy init. If anything this flatters mojodgl slightly. The
  harness was left untouched (per the change rule) and the effect is bounded
  by the ranges above. Timing is sequential (mojo first, then dgl) under
  `flock /tmp/mojo-bench.lock`, `gc.collect()` between cases.

## Tests — parity still asserted, nothing restored

- `git log -p -- tests/`: only two commits touch `tests/` in the full
  history — `973826d` (the port, initial test suite, 2026-10-02) and
  `95edaa7` (toolchain, pre-dates the port). `7f86f84` ("phase analyze")
  touches only `bench/analysis.md` and `bench/profile_deep.py`; `d75376e`
  ("phase verify", prior session) touches only `README.md`,
  `bench/_verify_parity.py`, `bench/verify.md`. `git diff 973826d HEAD --
  tests/` is empty.
- No removed assertions or loosened tolerances exist to restore. The suite
  still asserts parity with upstream everywhere (`assert_close` ->
  `rtol=1e-12` with `atol` 1e-9–1e-13 across `test_ops.py`,
  `test_function.py`, `test_edge_softmax_eids.py`, `test_parallel.py`,
  `test_graph.py`), including the parallel paths, every feature width 1–33,
  offset views, zero-length segments, and the 100-random-selection
  edge_softmax sweep. The only skips are `pytest.importorskip("torch"/"dgl")`
  guards in `tests/upstream.py` (skip only if upstream is not installed).
- `pixi run build && pixi run test`: 295 passed (see command log above).

## What changed in this phase

- **README** Performance table + surrounding prose: replaced all 16 rows with
  this session's four-invocation medians above; the "loses six rows / ties
  segment_reduce sum" prose corrected to the measured standing (loses five
  rows; `u_mul_e_sum` and `segment_reduce` sum at parity); machine/shape/
  method sentences updated to say median-of-four-in-this-session and to cite
  this file. No row was removed or reworded to look better;
  slower-than-upstream rows remain slower. The bold on the two largest wins
  (`segment_softmax`, `gather_mm`) is kept as in the previous table.
- **bench/verify.md**: this file (replaces the prior session's version).
- **No library code changed** (`src/`, `python/`, `tests/`, `bench/bench.py`,
  `pixi.toml`, `bench/_verify_parity.py` untouched). No tests were modified,
  so nothing to restore. No bug was shown with a failing test in this
  session, so no code fix was made.
- **Final gate**: `pixi run build && pixi run test` re-run after the README
  edit — see below.

## What could not be reproduced

- `pixi run bench` as literally written (guard bug, reported above). All
  bench evidence in this file is via the equivalent `pixi run bench-here`.
- Absolute numbers will move with box load (shared 72-core box, upstream is
  OpenMP-multithreaded, mojo kernels split over 32 workers); the *rows*
  that could not be made to flip are the four consistent losses above
  (`u_dot_e`, `edge_softmax` x3), which were slower than upstream in every
  one of the four invocations in this session. `u_mul_e_sum`,
  `u_add_e_mean`, `segment_reduce` sum, and `update_all` `u_mul_e` + `sum`
  flipped sign across invocations. Any single number in the corrected
  table should be read with its range.
