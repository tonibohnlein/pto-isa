# L0-tile cost-model study

A perf-sim study that validates the analytic cost model behind PyPTO's
`AutoTileMatmulL0` tile chooser (`utils::ChooseL0Tile`). It pins down, against the
pto-isa pipeline simulator, the numbers and the algorithmic choices a closed-form
L0 GEMM tile solver should make on **a2a3**.

Everything here builds under `-D__COSTMODEL` and runs the **host** pipeline
simulator — no NPU required. One command reproduces all of it:

```bash
python run.py            # generate -> build -> run -> analyze, all experiments
python run.py gemm_dbc   # a single experiment
python run.py --no-build # re-analyze existing CSVs
```

## Scope: L1 → L0 only

The tile chooser decides the **L0** GEMM tile `(m, n, k)` *after* a higher
(single-core / L1) tiling has already staged the operands into L1 / Mat. So this
study models the **L1 → L0** boundary and the cube, and deliberately treats the
`GM → L1` (MTE2) pipe as out-of-scope noise. That boundary is why these GEMMs are
**compute-bound, not memory-bound**: the operands are resident in fast L1 SRAM and
re-read with high arithmetic intensity. (Classic "GEMM is memory-bound" lives at
the `HBM/GM ↔ L1` boundary, handled by the level above this chooser.)

## a2a3 cost model

From `include/pto/costmodel/a2a3/...`:

| quantity | value |
| --- | --- |
| L1 → L0A bandwidth | **441 GB/s** (the cube's A / "left" port) |
| L1 → L0B bandwidth | **220.5 GB/s** (B / "right" port — exactly half) |
| L0C → L1 (FIXPIPE) | 128 GB/s   ·   L0C → GM | 70 GB/s |
| frequency | 1.85 GHz |
| MAD (one TMATMUL) | `6 + cpr·⌈m/16⌉·⌈k/kt⌉·⌈n/16⌉`, `kt = 32/bytes_a`, `cpr = 2 fp32 / 1 bf16` |
| L0A = L0B | 64 KiB (÷2 per slot when double-buffered) |
| L0C | 128 KiB (A5: 256 KiB) |

The **MTE1 pipe is shared** by L1→L0A and L1→L0B, so the two operand loads
serialize. The roofline (wall ≈ `max` over pipes) holds **only because the
operand loads are double-buffered** — load(i+1) overlaps compute(i). The lowering
provides that (ping-pong L0A/L0B), so the chooser may assume it.

## Roofline objective (validated here)

Minimize `max(C_load, C_mad)` over the L0-cap-constrained `(m,n,k)` grid
(`C_fix = M·N·bytes_c/BW_L0C` is a shape-independent floor, out of the argmin):

```
C_load(split-K) = M·N·K·( bytes_a/(n·BW_A) + bytes_b/(m·BW_B) )       # both operands streamed
C_load(full-K)  = stationary-panel reuse (min of A- / B-stationary, TIME-weighted)
C_mad           = (M/m)(N/n)(K/k) · ( 6 + cpr·⌈m/16⌉⌈k/kt⌉⌈n/16⌉ )
```

Memory-bound optimum aspect: `m:n = (bytes_b·BW_A)/(bytes_a·BW_B) = 2:1` for bf16
(tall — reload the slow-L0B operand less often).

## Experiments & findings

| testcase | kernel | tests | headline result |
| --- | --- | --- | --- |
| `gemm_sweep` | (reuses `gemm_performance`) | CUBE closed form + L0A/L0B asymmetry | CUBE exact 41/41; tall tile never loses to its transpose on MTE1 |
| `gemm_fullk` | `fullk_reuse_kernel.cpp` | full-K reuse + stationary choice | reuse cuts MTE1 **37–77%**; stationary winner = **bandwidth-weighted** `T_row/T_col` (10/10), aspect-dependent; **bytes-only mis-picks 2/10** (symmetric tiles) |
| `gemm_dbc` | `dbc_kernel.cpp` | L0C double-buffering | hiding the exposed FIXPIPE drain wins **13–27%** wall, even vs the larger single-buffer tile |
| `gemm_accblock` | `accblock_kernel.cpp` | **variant 3**: accumulator/C-blocking | `NACC` L0C accumulators give split-K A-reuse (MTE1 → ~B-only floor) without needing full-K |
| `gemm_asymbuf` | `asymbuf_kernel.cpp` | **variant 4**: asymmetric buffering | double-buffering only the *moving* operand cuts wall **6–9%** at identical MTE1 |

### Why bandwidth-weighting matters (gemm_fullk)

For a **symmetric** tile (e.g. `128×128`), A-stationary and B-stationary move
*identical bytes*, yet B-stationary uses ~33% less MTE1 — keep the slow-L0B
operand resident, reload the fast-L0A one. A bytes-only chooser ties here and can
pick the worse option. The winner flips with aspect (tall `m>n` → A-stationary),
and the sim follows the **time**-weighted formula on every config.

### The four algorithms (two used today, two recorded)

1. **split-K** — `k<K`, both operands streamed, no reuse (`gemm_sweep`, baselines).
2. **full-K** — `k==K`, one operand held stationary in L0, reused across the grid
   (`gemm_fullk`). Stationary side picked by time-weighted `T_row/T_col`.
3. **accumulator / C-blocking** — `NACC` L0C accumulators reuse the streamed A
   across `NACC` columns while still splitting K; decouples A-reuse width from the
   L0B cap (`NACC·m·n·bytes_c ≤ L0C`). Validated in `gemm_accblock`.
4. **asymmetric-buffered full-K** — stationary operand single-buffered so it uses
   the *full* L0 buffer (no ÷2), only the moving operand double-buffered. Same
   traffic, better overlap. Validated in `gemm_asymbuf`.

## Reproduce

`run.py` generates each `testcase/<name>/main.cpp`, builds it under `__COSTMODEL`,
runs it (per-kernel CSVs land in `results/perf_sim_output/`), and analyzes. The
`main.cpp` files are committed for convenience but are regenerated on every run.

```bash
python run.py                          # all experiments
python run.py --build-dir /path/to/bd  # reuse a cmake build dir
```

Needs `cmake`, a C++23 compiler, and GTest. The cost-model formula headers
(`*/formula_params_generated.hpp`) are auto-generated on first run.

## Layout

```
l0_tile_study/
  README.md            this file
  common.py            paths, a2a3 constants, MAD formula, CSV reader, codegen helpers
  experiments.py       the five experiment generators (config -> testcase main.cpp)
  analyze.py           per-experiment validation of the perf-sim CSVs
  run.py               generate -> build -> run -> analyze
  results/             generated CSVs + index json (git-ignored)
testcase/gemm_{sweep,fullk,dbc,accblock,asymbuf}/   kernels + generated main.cpp
```

## Caveats

- **MTE2 noise.** `gemm_dbc`/`gemm_asymbuf` totals include `GM→L1`, absent in the
  resident-operand case — so the wall-clock wins shown are **conservative**.
- **Store target.** Kernels store L0C→GM (70 GB/s); chained-matmul drains L0C→L1
  (128 GB/s). Magnitudes scale, the exposure/overlap conclusions hold.
- **a5** numbers are not yet measured; the eventual chooser parameterizes the cost
  model per arch (a2a3 real, a5 placeholder).
