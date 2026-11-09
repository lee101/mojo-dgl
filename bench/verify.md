# VERIFY phase — what was reproduced, what was not, what changed

Date: 2026-11-09. Machine for every number below:
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores, Linux x86_64, AVX2
(`W = 4`), torch 2.3.1.post100, dgl 2.3.0 (upstream 2.3.0 via pixi).
Box was shared throughout (load averages observed 9–77 across the session).

## Commands run (in order)

1. `pixi run build && pixi run test` — green. 295 passed in 9.23s.
2. `pixi run bench` — **does not run the harness.** It prints only:
   `run this through pixi run bench, which takes the machine-wide lock…`
   and exits 1. This is a wiring bug in the checkout: `pixi.toml`
   `bench = flock /tmp/mojo-bench.lock python bench/bench.py` holds the flock
   but never passes `--locked`, and `bench/bench.py:257-263` raises
   `SystemExit` unless `--locked` is in `sys.argv`. The flock-held invocation
   that actually executes the harness is `pixi run bench-here`
   (`flock /tmp/mojo-bench.lock python bench/bench.py --locked`; same lock,
   same harness, best-of-3 per case, `flock` held for the whole run). All bench
   numbers below use that invocation; `bench/analysis.md` reached the same
   conclusion independently. Only `bench/bench.py` and `pixi.toml` wire this
   guard; no correctness code is involved. It is reported here, not fixed,
   because the task said to change code only to fix a failing-test bug and
   `pixi run build && pixi run test` is unaffected.
3. `pixi run bench-here` × 4 (four separate invocations, each best-of-3 per
   case). Logs retained in this session at `/tmp/verify-bench*.log`;
   medians/ranges are computed from those four runs (table below).
4. `bench/_verify_parity.py` (new file, this phase): runs all 16 bench cases
   at the bench shape (200k nodes / 1M edges / 32d, same seeds as
   `bench/bench.py`) through both libraries on identical arrays and compares
   outputs. Result: **all 16 parity-OK** (see "Same input / no skipped work").

## README table vs measured — every row was outside 20%

The README table (158.7 ms mojo for `u_mul_e_sum`, 306.5 ms for `u_mul_v`,
380.3 ms for `segment_softmax`, …) did **not** reproduce. Every one of the
16 rows is outside the ±20% band on both columns, in most cases by 2–8×.
Comparing the README's mojo column to the verify median, mojo is 0.13–0.39×
the README figure (README overstates mojo time by roughly 2.5–8×); the dgl
column is 0.29–0.78× the README figure. No row was close enough to keep.

This is consistent with `bench/analysis.md`'s own two recorded runs (Run A/B),
whose mojo numbers (e.g. `u_mul_e_sum` 30.4/32.7 ms, `gather_mm` 30.2/22.7 ms)
are the same order as this verify session and far below the README table.
The README table appears to predate a performance change and was never
updated; it is not a measurement of the shipped code on this box.

### Corrected table (median of the four verify invocations)

Each invocation is best-of-3 per case (bench.py `timeit`, `repeat=3`);
the table is the median across the four invocations.

| case | mojo-dgl | dgl | speedup | per-run mojo range | per-run dgl range |
| --- | ---: | ---: | ---: | --- | --- |
| `u_mul_e_sum` (1M edges, 32d) | 28.2 ms | 27.4 ms | 0.97x | 26.1–30.7 | 25.3–35.1 |
| `u_add_e_mean` (1M edges, 32d) | 32.4 ms | 37.5 ms | 1.16x | 27.7–37.0 | 33.5–40.8 |
| `u_div_e_min` (1M edges, 32d) | 28.3 ms | 149.8 ms | 5.30x | 26.5–38.3 | 125.8–173.5 |
| `copy_u_max` (1M edges, 32d) | 19.0 ms | 39.1 ms | 2.07x | 17.2–24.6 | 33.9–48.5 |
| `copy_e_mean` (1M edges, 32d) | 26.2 ms | 30.9 ms | 1.18x | 20.5–33.3 | 29.0–34.2 |
| `u_dot_e` (SDDMM, 1M edges, 32d) | 21.8 ms | 10.6 ms | 0.49x | 21.1–25.8 | 9.5–11.3 |
| `u_mul_v` (SDDMM, 1M edges, 32d) | 40.4 ms | 47.2 ms | 1.17x | 37.5–63.8 | 40.4–56.8 |
| `edge_softmax` norm_by=dst (1M edges) | 10.8 ms | 2.4 ms | 0.22x | 10.6–11.7 | 2.1–2.8 |
| `edge_softmax` norm_by=src (1M edges) | 10.9 ms | 2.3 ms | 0.22x | 9.9–11.9 | 2.3–2.4 |
| `edge_softmax` 8 heads (1M edges, `(E, 8, 1)`) | 32.4 ms | 19.4 ms | 0.60x | 31.4–44.5 | 19.1–21.2 |
| `segment_reduce` sum (1M rows, 32d, 200k segments) | 19.2 ms | 19.2 ms | 1.00x | 18.2–28.7 | 15.8–19.5 |
| `segment_reduce` max (1M rows, 32d, 200k segments) | 39.2 ms | 60.0 ms | 1.53x | 29.8–42.6 | 57.0–61.9 |
| `segment_softmax` (1M rows, 32d, 200k segments) | 56.1 ms | 286.2 ms | 5.10x | 43.6–66.0 | 274.5–318.0 |
| `gather_mm` (1M rows, 16x16, 64 blocks) | 23.7 ms | 141.8 ms | 5.98x | 22.7–37.9 | 128.2–167.8 |
| `update_all` `u_mul_e` + `sum` (1M edges, 32d) | 33.5 ms | 24.0 ms | 0.72x | 27.9–37.4 | 23.4–24.5 |
| `update_all` `copy_u` + `max` (1M edges, 32d) | 36.9 ms | 54.1 ms | 1.47x | 34.2–37.3 | 51.4–54.6 |

Rows where the port is slower than upstream stay slower in the table, exactly
as measured: `u_mul_e_sum` (0.97x), `u_dot_e` (0.49x), `edge_softmax` dst
(0.22x), src (0.22x), 8 heads (0.60x), `update_all` `u_mul_e` + `sum`
(0.72x) — six rows — plus `segment_reduce` sum at 1.00x (a tie).
`u_mul_e_sum` swings 0.82x–1.50x across individual verify invocations
(dgl 25.3–49.0 ms on a shared box while mojo stays 26.1–32.7 ms), so it is
parity within noise, not a clear win; the median 0.97x is reported as
measured. `u_mul_v` is the noisiest row (mojo 37.5–63.8 ms); even its worst
verify invocation (63.8 vs 56.8 ms on a load-77 box) is near parity.

Old README → corrected (mojo / dgl / speedup), for the record:

| case | old | corrected |
| --- | --- | --- |
| `u_mul_e_sum` | 158.7 / 60.4 / 0.38x | 28.2 / 27.4 / 0.97x |
| `u_add_e_mean` | 179.0 / 70.6 / 0.39x | 32.4 / 37.5 / 1.16x |
| `u_div_e_min` | 205.8 / 234.5 / 1.14x | 28.3 / 149.8 / 5.30x |
| `copy_u_max` | 99.2 / 52.6 / 0.53x | 19.0 / 39.1 / 2.07x |
| `copy_e_mean` | 91.4 / 53.5 / 0.59x | 26.2 / 30.9 / 1.18x |
| `u_dot_e` | 114.6 / 34.6 / 0.30x | 21.8 / 10.6 / 0.49x |
| `u_mul_v` | 306.5 / 75.4 / 0.25x | 40.4 / 47.2 / 1.17x |
| `edge_softmax` dst | 32.6 / 4.0 / 0.12x | 10.8 / 2.4 / 0.22x |
| `edge_softmax` src | 32.6 / 4.6 / 0.14x | 10.9 / 2.3 / 0.22x |
| `edge_softmax` 8 heads | 200.8 / 28.9 / 0.14x | 32.4 / 19.4 / 0.60x |
| `segment_reduce` sum | 60.2 / 30.5 / 0.51x | 19.2 / 19.2 / 1.00x |
| `segment_reduce` max | 102.2 / 104.0 / 1.02x | 39.2 / 60.0 / 1.53x |
| `segment_softmax` | 380.3 / 479.2 / 1.26x | 56.1 / 286.2 / 5.10x |
| `gather_mm` | 127.4 / 402.2 / 3.16x | 23.7 / 141.8 / 5.98x |
| `update_all` `u_mul_e` + `sum` | 164.1 / 44.5 / 0.27x | 33.5 / 24.0 / 0.72x |
| `update_all` `copy_u` + `max` | 131.0 / 87.6 / 0.67x | 36.9 / 54.1 / 1.47x |

## Same input / no skipped work — checked

- **Same input, both sides.** `bench/bench.py:data()` builds one graph and its
  feature arrays (`u`, `v`, `x`, `e`, `z`, `zh`, all from `default_rng(0)`)
  and the pair `(g, ug)` — mojodgl gets them as numpy, upstream as
  `torch.as_tensor` of the *same* buffers. `segment_data()` (seed 11) and
  `mm_data()` (seed 12) are likewise shared. Every `CASES` lambda passes the
  identical arrays to both sides (verified by reading all 16 `CASES` defs:
  `t(a) = torch.as_tensor(a)`, zero-copy view). The `u_div_e_min` case
  shifts both `x` and `e` by `+4.0` before either side sees them — that
  shift is applied once, outside the timed lambda, to the shared arrays.
- **Outputs match at the bench shape.** `bench/_verify_parity.py` ran all 16
  cases at 200k nodes / 1M edges / 32d through both libraries and compared:
  all 16 `allclose(rtol=1e-9, atol=1e-9)` = True, max abs diff ≤ 5.4e-13
  wherever both sides are finite. Two cases (`u_div_e_min`, `copy_u_max`)
  report `maxabsdiff=nan` in the script because 43,232 entries are `+inf` on
  *both* sides (zero-in-degree rows — upstream's `min`/`max` of an empty set)
  and `inf − inf = nan` in the diff; `allclose` is True and an `equal_nan`
  recheck confirms identical inf positions on both sides. No case silently
  computes less on one side: shapes match exactly (`(200000,32)` for gspmm
  cases, `(1000000,1)` for `u_dot_e`, `(1000000,8,1)` for 8-head softmax,
  etc.), and the update_all cases run the full message+reduce on both sides.
- **No caching that skips work.** `data()`/`segment_data()`/`mm_data()` use a
  module-level `_CACHE` so the *inputs* are built once — that is input
  sharing, not result caching; each timed rep calls the public op fresh.
  `update_all` outputs are recomputed each rep.
- **One asymmetry worth recording (not a skip, not corrected).** `main()`
  warms mojodgl once (`ours()`) before timing but never warms upstream
  (`theirs()`). Best-of-3 mostly absorbs this, but the first upstream rep may
  carry lazy init. If anything this flatters mojodgl slightly. The harness
  was left untouched (per the change rule) and the effect is bounded by the
  ranges above. Timing is sequential (mojo first, then dgl) under
  `flock /tmp/mojo-bench.lock`, `gc.collect()` between cases.

## Tests — parity still asserted, nothing restored

- `git log -p -- tests/`: only two commits touch `tests/` —
  `973826d` (the port, initial test suite) and `95edaa7` (toolchain, only
  touches `tests/` as the pre-existing tree). The one commit after the port,
  `7f86f84` ("phase analyze"), touches **only** `bench/analysis.md` and
  `bench/profile_deep.py` — `git show --stat HEAD` confirms zero changes to
  `tests/`, `bench/bench.py`, or any library code. `git diff 973826d HEAD --
  tests/` is empty.
- No removed assertions or loosened tolerances exist to restore. The suite
  still asserts parity with upstream everywhere (`assert_close` → `rtol=1e-12`
  with `atol` 1e-9–1e-13 across `test_ops.py`, `test_function.py`,
  `test_edge_softmax_eids.py`, `test_parallel.py`, `test_graph.py`), including
  the parallel paths, every feature width 1–33, offset views, zero-length
  segments, and the 100-random-selection edge_softmax sweep. The only skips
  are `pytest.importorskip("torch"/"dgl")` guards in `tests/upstream.py`
  (skip only if upstream is not installed, as the README documents).
- `pixi run build && pixi run test`: 295 passed (see command log above).

## What changed in this phase (verify only, per the change rule)

- **README** Performance table + surrounding prose: replaced all 16 rows with
  the four-invocation medians above; the "mojodgl still loses most of the
  table" prose corrected to the measured standing (loses 6 rows + 1 tie,
  wins 9); the `u_mul_e_sum` parity-within-noise note added. No row was
  removed or reworded to look better; slower-than-upstream rows remain
  slower. Machine/shape/method sentences updated to say median-of-four and
  to cite this file.
- **bench/verify.md**: this file.
- **bench/_verify_parity.py**: the parity checker described above (new
  helper, not part of the build/test path; it does not affect the shipped
  library or test suite).
- **No library code changed** (`src/`, `python/`, `tests/`, `bench/bench.py`,
  `pixi.toml` untouched). No tests were modified, so nothing to restore.
- **Final gate**: `pixi run build && pixi run test` re-run after the README
  edit — see below. (README edit cannot affect the build/test outcome, but
  the gate was re-run as required.)

## What could not be reproduced

- The original README numbers themselves: see above — all 16 rows failed the
  20% check, so none survive in the README.
- `pixi run bench` as literally written (guard bug, reported above). All
  bench evidence in this file is via the equivalent `pixi run bench-here`.
- Absolute numbers will move with box load (shared 72-core box, upstream is
  OpenMP-multithreaded, mojo kernels split over 32 workers); the *rows* that
  could not be made to flip are the six losses above, which were slower than
  upstream in every one of the four verify invocations
  (`u_dot_e`, `edge_softmax` ×3, `update_all u_mul_e + sum`) plus the
  `segment_reduce` sum tie. `u_mul_e_sum` flipped sign across invocations.
  Any single number in the corrected table should be read with its range.
