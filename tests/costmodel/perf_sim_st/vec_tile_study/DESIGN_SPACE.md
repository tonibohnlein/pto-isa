# Vector-core design space + mlsys26 cross-check

The mechanisms the mlsys26 vector model (`Ascend910BCost` vector path,
`num_vector_cores=48`) currently has, their pto-isa grounding, and the perf-sim
experiments that verify them. Sibling of `gm_l1_tile_study/DESIGN_SPACE.md`.

## Mechanisms inventory

| mechanism | mlsys26 | grounded? | experiment |
| --- | --- | --- | --- |
| per-op compute `slope·repeat + startup` | `head+slope·repeat+tail` per op | **form ✓, accounting ✗** | `vec_pointwise` [M] |
| per-op slope diversity | single `slope_pw=2` | ✗ (div=4, cheap=1) | `vec_pointwise` [M] |
| once-per-stream startup | per-op head+tail | ✗ (overcounts chains) | `vec_pointwise` [M] |
| count-mode floor (+16) | not modeled | ✗ | _planned_ |
| reduction slope (`vreducev2`) | `slope_reduce=14` | matches stub (stub itself uncalibrated) | _planned_ `vec_reduce` |
| reduced-axis sink split-S | the cube split-K analog, sink-only | — | _planned_ `vec_splitS` |
| implicit streaming recompute | `N_passes = #reductions+1` (upper bound) | ✗ | _planned_ `vec_stream` |
| GM↔UB per-direction `par()` cap | `io/par(active, bw)` | read pool ✓ (`gml1_contention`) | — |
| DMA-shape penalty (sub-burst width) | `max(1, burst/(w·dtype))` | — | _planned_ |
| double-buffer floor (small-tile serialize) | `max(c,d)` iff `tile≥2·reg` | — | _planned_ |

## What the perf-sim confirms (measured) [M]

| model claim | experiment | verdict |
| --- | --- | --- |
| vector op = `slope·repeat + head+tail` (isolated) | `vec_pointwise` | slope & startup match the device stub to **0.0%** (add 2/24, mul 2/25, div 4/30, exp 2/31) |
| startup paid **once per stream**, not per op | `vec_pointwise` | `vec = head+tail + NOPS·slope·repeat` to 0.0%; mlsys26's per-op charge overcounts **1.2×→2.7×** (NOPS 1→16) |

## vec_pointwise — the per-op formula + chain accounting [M]

`EstimateLinearCycles` (`cce_costmodel_core.hpp:133`) charges `head+tail` only when the
VECTOR pipe queue is empty — i.e. once at the start of a back-to-back vector stream. The
chain sweep (op=add, repeat=8) measures it directly:

| NOPS | sim `vec` | `head+tail + NOPS·slope·repeat` | mlsys26 `NOPS·(head+slope·repeat+tail)` | overcount |
| - | --- | --- | --- | --- |
| 1 | 40 | 40 | 48 | 1.2× |
| 4 | 88 | 88 | 192 | 2.2× |
| 16 | 280 | 280 | 768 | 2.7× |

**Implication for mlsys26.** The vector compute model is right in *form* (linear in
`repeat`, slope 2 pw / 14 reduce) but wrong in *accounting*: it charges the per-op
startup `head+tail` for every op, so a fused vector chain (softmax = exp+reduce+div in one
UB stream) is overcosted by `~(NOPS-1)·(head+tail)`. The fix is to charge the chain's
startup **once** (`total = startup + Σ slope·repeat`), not per op — and to use the
per-op slope (`vdiv`=4, `vmuls/vrelu`=1) rather than a single `slope_pw=2`.

## Next experiments

- **`vec_reduce`** — `vreducev2`/`vcadd` slope (14/7) + the row vs col reduced axis. Note
  `vreducev2` is the one op *not* device-calibrated in the stub, so measure it directly.
- **`vec_splitS`** — the sink-only reduced-axis cross-core split (the cube split-K analog):
  S partials, the thin `[H,1]`/`[1,W]` atomic-add store folded into the roofline.
- **`vec_stream`** — UB-overflow streaming: is the real recompute cost `#reductions+1`
  passes, or less (per-op liveness)?

## Caveats

- `vec_pointwise` keeps `COLS` VL-aligned (multiple of 64 fp32) so the fast path is taken
  (`repeat = ROWS·COLS/64`, one fused call). Unaligned widths fall into count-mode, which
  the stub charges as a flat `head+tail+16` (degenerate) — measured separately later.
- fp32 throughout; fp16 (`elements/repeat=128`) halves `repeat` for the same shape.
- Single-core (`blockDim=1`) — the per-op compute formula; the 48-core fill, reduced-axis
  split, and GM↔UB `par()` cap are separate (multi-core) experiments.
