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
| `gemm_dbc` | `dbc_kernel.cpp` | L0C double-buffering | hiding the exposed FIXPIPE drain wins **13–37%** wall (resident operands), even vs the larger single-buffer tile |
| `gemm_accblock` | `accblock_kernel.cpp` | **variant 3**: accumulator/C-blocking | `NACC` L0C accumulators give split-K A-reuse (MTE1 → ~B-only floor) without needing full-K |
| `gemm_asymbuf` | `asymbuf_kernel.cpp` | **variant 4**: asymmetric buffering | double-buffering only the *moving* operand cuts wall **6–9%** at identical MTE1 |

### Why bandwidth-weighting matters (gemm_fullk)

For a **symmetric** tile (e.g. `128×128`), A-stationary and B-stationary move
*identical bytes*, yet B-stationary uses ~33% less MTE1 — keep the slow-L0B
operand resident, reload the fast-L0A one. A bytes-only chooser ties here and can
pick the worse option. The winner flips with aspect (tall `m>n` → A-stationary),
and the sim follows the **time**-weighted formula on every config.

### The design space — see `DESIGN_SPACE.md`

The "variants" we measured are **not distinct algorithms** — they are settings on
**three** orthogonal axes that compose freely (per the BLIS loops-around-the-
micro-kernel form): **(1) tile sizes per level** — L0 block + accumulator micro-tile
`N_acc`; **(2) loop order → operand stationarity**; **(3) per-buffer pipeline
depth**. `DESIGN_SPACE.md` is the canonical decomposition (with cited decision
rules); each experiment exercises one axis:

| experiment | axis exercised |
| --- | --- |
| `gemm_sweep` | 1 — tile size / aspect (CUBE + L0A/L0B asymmetry) |
| `gemm_fullk` | 2 — operand stationarity (output- vs A- vs B-stationary) |
| `gemm_accblock` | 1 — accumulator micro-tile `N_acc` (= `mr×nr`, the finer level) |
| `gemm_dbc`, `gemm_asymbuf` | 3 — pipeline depth (L0C; moving-operand) |

### Regimes — when each shines (`compare.py`)

Cross-comparing split-K / full-K / accumulator-blocking at matched tiles
(resident operands) shows the reuse algorithms cut MTE1 hard — full-K deepest
(k==K), accblock between, split-K the floor — **but on these 512² tiles every
case is FIXPIPE-bound** (`fixp ≈ 12.9k` ≫ `mte1, cube`). So the MTE1 saving is
*free headroom, not speedup*: the wall is set by the drain. The lesson is regime-
dependent:

| regime | bound | what wins |
| --- | --- | --- |
| skinny / large-K (little drain, heavy reload) | **MTE1** | operand **reuse** (full-K / accblock) |
| small-K, large output (slow drain) | **FIXP** | **L0C double-buffering** (hide the drain) |
| large-K square | **CUBE** | the **biggest tile** (least head); reuse is headroom |

So no single algorithm dominates — the chooser must pick by the predicted bound.

### Always double-buffer A/B/C? No.

- **Moving** operands: yes (overlap load with compute — the roofline assumption).
- **Stationary** operand (full-K): **no** — single-buffer it so it uses the full
  L0 buffer (variant 4); double-buffering a held panel only wastes capacity.
- **L0C**: double-buffer when the drain is *exposed* (single-L0C stalls the cube
  per drain) and the tile still fits L0C/2 — worth 13–37% here, most in the
  FIXP-bound regime; not worth halving C0 when the drain is already hidden.

## References & standard terminology

These algorithms are **textbook** — none is a new tiling scheme. Map to the canon:

| our variant | literature name | cite |
| --- | --- | --- |
| 1. "split-K" (serial-K accum.) | **output-stationary** blocked GEMM | Lam-Rothberg-Wolf ASPLOS'91; Eyeriss ISCA'16 |
| 2. full-K operand-stationary | Goto **GEBP block-panel**; **weight/input-stationary** | Goto & van de Geijn TOMS'08; Eyeriss'16; TPU ISCA'17 |
| 3. accumulator / C-blocking | **register tiling** / rank-k with C resident (`N_acc` = `nr`) | Goto'08 §6.1; Smith et al. IPDPS'14; ATLAS'01 |
| 4. double-buffering | **multistage / software-pipelined** mainloop | CUTLASS efficient_gemm; TPU "+1 for double-buffering" |

Key URLs: Goto'08 `cs.utexas.edu/~flame/pubs/GotoTOMS.pdf` · BLIS many-threaded
`.../blis3_ipdps14.pdf` · CUTLASS `github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md`
· Eyeriss `people.csail.mit.edu/emer/media/papers/2016.06.isca.eyeriss_architecture.pdf`
· Stream-K `arxiv.org/abs/2301.03598`.

**⚠ Naming:** our **"split-K" is a misnomer** — in CUDA/CUTLASS, *split-K* means
partitioning K across *parallel* workers that then *reduce* partials (atomics or a
reduction kernel). Ours has no parallel reduction; it is **serial-K accumulation**.
Worth renaming in pass + study to avoid confusion.

**Not yet considered** (mostly *above* the L0 boundary): parallel split-K +
cross-core/atomic reduction, **Stream-K** (load-balanced K, fixes wave
quantization), threadblock **rasterization/swizzle** (L2 reuse), **producer/
consumer (TMA-style) pipelining** and **multi-stage `kStages>2`** buffering (richer
variant 4, genuinely at L0), persistent kernels, sliced-K. Strassen/sub-cubic is a
poor fit for a fixed-tile cube. The genuine novelty, if any, is **the per-shape
cost-model solver** that picks among these — not the primitives.

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
  README.md            this file (experiments + findings)
  DESIGN_SPACE.md      the untangled design space: orthogonal axes + decision rules
  common.py            paths, a2a3 constants, MAD formula, CSV reader, codegen helpers
  experiments.py       the five experiment generators (config -> testcase main.cpp)
  analyze.py           per-experiment validation of the perf-sim CSVs
  compare.py           cross-comparison: split-K vs full-K vs accblock at matched tiles
  run.py               generate -> build -> run -> analyze
  results/             generated CSVs + index json (git-ignored)
testcase/gemm_{sweep,fullk,dbc,accblock,asymbuf}/   kernels + generated main.cpp
```

## Caveats

- **Operands are modeled L1-resident** (the custom kernels issue no `GM→L1`
  `TLOAD`, only `L1→L0` `TEXTRACT`), so MTE2 is excluded *by construction* — the
  faithful L1→L0 scope. `gemm_sweep` reuses the reference kernel (which does load)
  but is read on MTE1 only, so it is unaffected.
- **Store target (the live caveat).** Kernels drain L0C→GM (70 GB/s), which makes
  these 512² problems FIXPIPE-bound; chained-matmul drains L0C→L1 (128 GB/s, ~half
  the FIXP), shifting the bound toward CUBE/MTE1. The exposure/overlap conclusions
  hold; the *regime* read depends on the drain target.
- **a5** numbers are not yet measured; the eventual chooser parameterizes the cost
  model per arch (a2a3 real, a5 placeholder).
