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
| DMA-shape penalty (sub-burst width) | `max(1, burst/(w·dtype))` | **perf-sim shape-blind — can't validate** (device-eval only) | `vec_dma` [M] |
| double-buffer floor (`max` vs serialize) | `max(c,d)` iff `tile≥2·reg` | naive serializes (`t/sum=1`); software-pipelined overlaps (`t/sum→0.55`) — `max` needs `SetFlag/WaitFlag` | `vec_dma` [M] |
| dtype scaling (`epr=reg/dtype_bytes`) | `repeat = elems/epr` | **✓** | `vec_fp16` [M] |

## What the perf-sim confirms (measured) [M]

| model claim | experiment | verdict |
| --- | --- | --- |
| vector op = `slope·repeat + head+tail` (isolated) | `vec_pointwise` | slope & startup match the device stub to **0.0%** (add 2/24, mul 2/25, div 4/30, exp 2/31) |
| startup paid **once per stream**, not per op | `vec_pointwise` | `vec = head+tail + NOPS·slope·repeat` to 0.0%; mlsys26's per-op charge overcounts **1.2×→2.7×** (NOPS 1→16) |
| a reduction is a barrier-separated **tree**, not one op | `vec_reduce` | `TROWSUM = 45·(COLS/64)+6`, `TCOLSUM = 16(R-1)+30·log₂R`, both to **0.0%** |
| reduce-W (`TROWSUM`) is **ROWS-independent** (count mode) | `vec_reduce` | sim flat at 96 over ROWS 8→64; mlsys26 (`repeat=ROWS·COLS/64`) overcounts up to **19×** |
| UB-overflow streaming is **online** (~1 wide pass), not `#reductions+1` | `vec_stream` | online softmax wide body flat (≤1×) over NCHUNKS 1→8; mlsys26's `3×` is 3–4× over |
| reduced-axis split-S: per-core reduce ∝ `Wc`, merge = `S·[H,1]` | `vec_splitS` | per-core 366→51 (S 1→8) to 0.0%; `[H,1]` store a 3-cycle floor; merge `S·store` |
| GM↔UB cost is shape-blind (no sub-burst-width penalty) | `vec_dma` | `mte2` flat at 8928 / 101 GiB/s over W 16→512 (0.0% spread) — the perf-sim charges by total bytes |
| roofline: naive serializes, software-pipelined overlaps | `vec_dma` | naive `t/sum=1.00`; pipelined (`SetFlag/WaitFlag` prefetch) `t/sum→0.55`, `t/max→1.36` |
| dtype scaling `epr=reg/dtype_bytes` | `vec_fp16` | half (`epr=128`) halves `repeat` + reduce-`K`; pointwise & reduce match to 0.0% |

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

## vec_dma — GM↔UB I/O: shape penalty + roofline overlap [M]

Two GM↔UB terms in the vector roofline were "reasoned bounds, not measured" — `vec_dma`
grounds both, and the answer for each is "the perf-sim is coarser than mlsys26 / real HW":

- **DMA-shape penalty.** Loading a fixed-byte tile at widths W = 16→512 gives the **same
  `mte2` (8928, 101 GiB/s, 0.0% spread)** — the perf-sim charges GM↔UB by `nBurst·lenBurst` =
  total bytes, **shape-blind**. So mlsys26's `max(1, vec_reg_bytes/(w·dtype))` (sub-burst
  widths cost more) is a real-HW effect the perf-sim **cannot validate**; keep it a
  **device-eval reasoned bound**, not perf-sim-grounded. (The model isn't wrong — the
  perf-sim just can't confirm or refute it.)
- **Roofline overlap.** A naive single-buffer `load→compute→store` loop has `t/sum = 1.00`
  exactly — it **serializes**. A **software-pipelined** kernel (prefetch tile s+1 with
  `SetFlag/WaitFlag` while computing tile s) **overlaps** the GM↔UB DMA with VEC: `t/sum`
  drops 1.00 → **0.55**, `t/max` → 1.36 (approaching 1 in steady state). So the
  `max(compute, ddr)` roofline **is** achievable — but *only* with explicit pipelining;
  buffer-alternation alone is **not** enough (the perf-sim sums program-order ops). mlsys26's
  `tile ≥ 2·vec_reg_bytes ⇒ max` is the right *shape*, **conditioned on the emit software-
  pipelining** (as the cube gemm does, per `gml1_roofline`) — not automatic from tile size.

**`vec_fp16`** closes the dtype assumption: half (`epr = 256/2 = 128`) halves the pointwise
`repeat` and the reduce tree `K`; both match `VecOpCompute` to **0.0%**.

## The vector cost model — grounded fixes for mlsys26

All four experiments point one way: **the mlsys26 vector model is systematically
pessimistic** — it overcharges every mechanism (1.2× → 19×), so softmax / layernorm /
attention vector stages look far more expensive than the perf-sim says, distorting fusion
decisions. The grounded fixes (in `3rdparty/mlsys26/src/core/ascend910b_cost.cpp`,
constants perf-sim-grounded + device-eval-pending):

| # | finding | fix | status |
| - | --- | --- | --- |
| **1** | reduction `repeat=ROWS·COLS` overcounts up to **19×** | cost a reduction by its **reduced-axis tree** (`VecOpCompute`): reduce-W `45·(K-1)+51` (ROWS-independent), reduce-H `16·(H-1)+30·log₂H` | **done** — shared by the vector-only + mixed paths |
| **2** | UB-overflow `×(#reductions+1)` overcounts **3–4×** | **online** model: wide body ×1, IO read once, only a thin `O(nchunks·#red)` per-chunk surcharge | **done** — the stale `STREAM >4×` test updated to the grounded `~linear` scaling |
| **3** | per-op startup overcounts fused chains **1.2–2.7×** | charge `head+tail` **once per back-to-back stream** (reductions/matmuls break it), not per op | **done** — `pw_stream_start` tracked across the op chain in both paths |
| **4** | split-S `compS` divides the broken reduction cost | follows from Fix 1 (`eval_reduce_S` now divides the corrected cost) | **done** via Fix 1 |

Net: vector stages get **cheaper and correctly-shaped**. The reductions (Fix 1) flip tall
row-reduces from spuriously compute-bound to DDR-bound; streamed softmax (Fix 2) no longer
carries a phantom 3× recompute. `vec_tile_study` is the regression oracle — every formula
matches the perf-sim to 0.0% today, so re-run it after any coefficient change.

**Device-eval caveat (applies to all):** the perf-sim is the measured ground truth here,
but its count-mode flat-per-pass (Fix 1) and online-streaming (Fix 2) are themselves coarse
vs real HW — the constants (`45/51/16/30`, the streaming surcharge) are pending the device
evaluation, same as the cube-side HBM-900.

## Next experiments

The single-core vector model is now comprehensively grounded (compute, reductions,
streaming, split, GM↔UB I/O + the roofline overlap, dtype). The one remaining piece is
separate:

- **Mixed cube+vector** — the `sat≈1` DDR wall + cube↔vector HBM contention (the M1/M2/M3
  plan). A separate `mixed_*` study, not a `vec_tile_study` expansion.

## Caveats

- `vec_pointwise` keeps `COLS` VL-aligned (multiple of 64 fp32) so the fast path is taken
  (`repeat = ROWS·COLS/64`, one fused call). Unaligned widths fall into count-mode, which
  the stub charges as a flat `head+tail+16` (degenerate) — measured separately later.
- fp32 throughout; fp16 (`elements/repeat=128`) halves `repeat` for the same shape.
- Single-core (`blockDim=1`) — the per-op compute formula; the 48-core fill, reduced-axis
  split, and GM↔UB `par()` cap are separate (multi-core) experiments.
