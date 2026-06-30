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
| count-mode floor (+16) | not modeled | ✗ | `vec_reduce` [M] (reductions run in count mode) |
| reduction cost | `head+slope_reduce·repeat+tail`, `repeat=ROWS·COLS/64` | **✗ — structurally wrong** (a tree, not one op; reduce W is ROWS-independent) | `vec_reduce` [M] |
| reduced-axis sink split-S | the cube split-K analog, sink-only | per-core reduce ✓; merge = S·[H,1] | `vec_splitS` [M] |
| implicit streaming recompute | `N_passes = #reductions+1` (upper bound) | **✗ — 3–4× pessimistic** (real emit is online) | `vec_stream` [M] |
| GM↔UB per-direction `par()` cap | `io/par(active, bw)` | read pool ✓ (`gml1_contention`) | — |
| DMA-shape penalty (sub-burst width) | `max(1, burst/(w·dtype))` | — | _planned_ |
| double-buffer floor (small-tile serialize) | `max(c,d)` iff `tile≥2·reg` | — | _planned_ |

## What the perf-sim confirms (measured) [M]

| model claim | experiment | verdict |
| --- | --- | --- |
| vector op = `slope·repeat + head+tail` (isolated) | `vec_pointwise` | slope & startup match the device stub to **0.0%** (add 2/24, mul 2/25, div 4/30, exp 2/31) |
| startup paid **once per stream**, not per op | `vec_pointwise` | `vec = head+tail + NOPS·slope·repeat` to 0.0%; mlsys26's per-op charge overcounts **1.2×→2.7×** (NOPS 1→16) |
| a reduction is a barrier-separated **tree**, not one op | `vec_reduce` | `TROWSUM = 45·(COLS/64)+6`, `TCOLSUM = 16(R-1)+30·log₂R`, both to **0.0%** |
| reduce-W (`TROWSUM`) is **ROWS-independent** (count mode) | `vec_reduce` | sim flat at 96 over ROWS 8→64; mlsys26 (`repeat=ROWS·COLS/64`) overcounts up to **19×** |
| UB-overflow streaming is **online** (~1 wide pass), not `#reductions+1` | `vec_stream` | online softmax wide body flat (≤1×) over NCHUNKS 1→8; mlsys26's `3×` is 3–4× over |
| reduced-axis split-S: per-core reduce ∝ `Wc`, merge = `S·[H,1]` | `vec_splitS` | per-core 366→51 (S 1→8) to 0.0%; `[H,1]` store a 3-cycle floor; merge `S·store` |

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

## vec_reduce — a reduction is a barrier-separated tree, not one op [M]

`TROWSUM` (reduce W, `[H,W]→[H,1]`) lowers to a binary tree of `vadd` passes + a final
`vcadd`, **each separated by `pipe_barrier(PIPE_V)`** — which flushes the VEC queue
(`trace.hpp` `queue.clear()`), so every pass re-pays `head+tail`. The passes run in
**count mode** (`repeat=0`), so `slope·repeat` vanishes and the cost is pure per-pass
startup. Measured == predicted to 0.0%:

| | formula | scales with | mlsys26 overcount |
| --- | --- | --- | --- |
| `TROWSUM` (reduce W) | `45·(COLS/64) + 6` | **COLS only** (ROWS-independent) | 2.5× → **19× on tall tiles** |
| `TCOLSUM` (reduce H, binary) | `16(R-1) + 30·log₂R` | **ROWS** (the reduced dim) | 1.3–1.5× |

**Implication for mlsys26.** The reduction model — one op, `head + slope_reduce·repeat +
tail` with `repeat = ROWS·COLS/64` — is **structurally wrong**. The real cost scales with
the **reduced dimension's tree**, not `ROWS·COLS`: reducing W is ROWS-independent (so a
`[64,128]` row-reduce is overcounted 19×), reducing H scales with H. The fix is to cost a
reduction by its reduced axis (`~k·(W/64)` for a row-reduce, `~k'·H` for a col-reduce),
not the product. **Caveat:** the perf-sim's count-mode flat-per-pass is itself coarse vs
real HW (a count-mode op over more rows *does* cost more on device) — flag for the device
eval; here it is the measured ground truth the analytic model must match.

## vec_stream — UB-overflow streaming is online (~1 pass), not #reductions+1 [M]

When a reduced band overflows UB the schedule streams the reduced axis in chunks. mlsys26
multiplies compute + IO by `N_passes = #reductions + 1` (softmax = 3) and flags it a
"pessimistic upper bound." The canonical pto-isa emit (`pto_macro_fa_softmax`) is **online
/ flash**: each chunk's `exp` runs **once per element**; a running max/sum is corrected with
a thin `[H,1]` rescale per chunk — no re-read. Measuring a softmax numerator (rowmax →
center → exp → rowsum) over `[8,512]` streamed over COLS in NCHUNKS:

| NCHUNKS | chunk W | `vec_cycles` | ratio to materialized | mlsys26 `3×` |
| - | --- | --- | --- | --- |
| 1 | 512 | 967 | 1.00× | 3.0× over |
| 4 | 128 | 940 | 0.97× | 3.1× over |
| 8 | 64 | 728 | 0.75× | 4.0× over |

The wide body is **flat** in NCHUNKS (the per-chunk reductions re-split the *same* total
work; at chunk W=64 they hit the cheaper single-`vcadd` base, so it even dips). The measured
kernel omits the thin flash correction (a layout-incompatible `[H,1]` `TEXP`/`TMAX`), but
that surcharge is `O(NCHUNKS)` thin-tile ops — small. So the real online cost is `~1× wide
body + O(NCHUNKS) thin`, and mlsys26's `3×` overcounts every streamed softmax by **3–4×**.

**Implication for mlsys26.** Replace the `#reductions+1` multiplier with what its own comment
asks for: a per-op-liveness model — wide-body recompute factor **≈1**, plus a per-chunk
surcharge (re-paid vector startup at each barrier + `O(NCHUNKS·ROWS·1)` thin correction). The
current model triples the cost of the large-context attention regime it most needs to get right.

## vec_splitS — the reduced-axis cross-core split (cube split-K analog) [M]

A sink reduction over `[H,W]` split S ways: each of S cores reduces its band `[H, Wc=W/S]`
→ `[H,1]` and the S partials atomic-add merge. Mirroring `gml1_splitk`, we measure the
**per-core subproblem** on one core (the perf-sim is per-core; the cross-core merge is a
cost-model term). Per-core `vec` matches the reduction formula to 0.0% and drops with Wc;
the `[H,1]` partial store (`mte3`) is a constant floor:

| S | Wc | per-core `vec` | `[H,1]` store | merge `S·store` |
| - | --- | --- | --- | --- |
| 1 | 512 | 366 | 3 | 3 |
| 4 | 128 | 96 | 3 | 12 |
| 8 | 64 | 51 | 3 | 24 |

So split-S trades **per-core compute (~1/S)** for **merge (~S thin partials)** — the
`eval_reduce_S` tradeoff, where an optimal S balances the two and the bound flips
compute→merge at a knee. **But** the per-core reduce inherits the `vec_reduce` finding: it's
ROWS-independent and tracks the reduced-axis tree, whereas mlsys26's `compS` divides the
*wrong* `total_compute` (`slope_reduce·ROWS·COLS`) — so the split decision is built on a
reduction cost that's already up to 19× off. Fix the reduction cost first, then the split.

(Caveat: at the exact shapes `H·W = 8192` with a ColMajor `[H,1]` dst — `8×1024`, `16×512`,
`32×256`, `64×128` — the reduce takes a faster `vcgadd` `TryOptimizeFP32Reduce` path, ~2.5×
cheaper than the generic tree; `vec_splitS` avoids those shapes so the per-core trend is clean.)

## Next experiments

- **`vec_dma`** — GM↔UB I/O: the DMA-shape penalty (sub-burst width) + the double-buffer
  serialization floor, on the `mte2_aiv`/`mte3` pipes.

## Caveats

- `vec_pointwise` keeps `COLS` VL-aligned (multiple of 64 fp32) so the fast path is taken
  (`repeat = ROWS·COLS/64`, one fused call). Unaligned widths fall into count-mode, which
  the stub charges as a flat `head+tail+16` (degenerate) — measured separately later.
- fp32 throughout; fp16 (`elements/repeat=128`) halves `repeat` for the same shape.
- Single-core (`blockDim=1`) — the per-op compute formula; the 48-core fill, reduced-axis
  split, and GM↔UB `par()` cap are separate (multi-core) experiments.
