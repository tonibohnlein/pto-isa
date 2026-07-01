# Mixed cube+vector design space & the mlsys26 cross-check

The mixed kernel is a matmul (cube/AIC) feeding a pointwise/reduction epilogue (vector/AIV)
that hands off **only through GM** (no UB↔Mat/L1 path on the 910b). The scheduler's job is to
**hide the cube and vector compute behind each other and behind the GM traffic** by
software-pipelining the two units. This study grounds the analytic model of that overlap
against the perf-sim. See `README.md` for the headline numbers; this file is the design-space
map and the mlsys26 model cross-check.

## The axes

| axis | values | who decides | effect on the wall |
| --- | --- | --- | --- |
| **schedule** | serial (no skew) ↔ skewed (producer one tile ahead) | `SkewCrossCorePipeline` | serial = `cube+vec` (or worse); skewed = `max(cube,vec)+fill` |
| **tile count** | NTILES (the producer-skew depth amortizer) | the output tiling | fill = one cube tile, constant → its *share* is `1/NTILES`-ish |
| **cube:vector balance** | matmul K / epilogue op-count | the fused group | which stage the `max()` picks |
| **GM round-trip** | cube store + vec load + vec store | the handoff buffering | the `ddr` floor the compute must beat |

This study nails the **schedule** and **tile-count** axes; the **balance** and **GM** axes are
the next experiments (`mixed_balance`, `mixed_ddr_bound` — see "Next").

## What the perf-sim proves about the schedule axis

The perf-sim co-simulates AIC and AIV on **one event clock**; `total_cycles` is the wall.
Overlap is the default — pipes run concurrently unless a **RAW** data dep links them (the
`TileDepTracker` matches GM addresses; WAR/WAW are not tracked, and no FFTS is needed). So the
schedule is encoded entirely in the handoff buffering:

- **skewed (ping-pong)** — cube(k+1) writes the *other* buffer than vector(k) reads → no dep →
  the CUBE pipes (mte2_aic/mte1/cube/fixp) run concurrently with the VEC pipes
  (mte2_aiv/vec/mte3). Wall → `max(cube, vec) + fill`, fill = the one-tile prologue.
- **serial (RAW chain)** — the next cube's operand comes from the buffer the prior vector
  wrote → cube(k) waits vector(k−1). Wall → `cube + vec`, and **beyond** it, because isolating
  each tile behind the handoff *also* serializes the cube's own cross-tile pipeline.

Ground truth (bm=128, N=128, K=128), `t/pipe = total/(max+fill)`, `t/srl = total/(cube+vec)`:

```
                 overlap run                 serial run
NT  cubeW vecW  total t/pipe ovl |  total t/srl t/pipe ovl
 1  2426  1762  4188   0.86 0.00 |  4188   1.00  0.86  0.00
 2  3751  3063  5489   0.89 0.43 |  7925   1.16  1.28  0.00
 4  6401  5713  8139   0.92 0.70 | 15399   1.27  1.74  0.00
 8 11701 11013 13439   0.95 0.84 | 30347   1.34  2.15  0.00
```

The same work runs **2.26× faster** skewed than serial at NT=8 — the value the cross-core
pipeline delivers, and the value `SkewCrossCorePipeline`'s demote-to-sequential path leaves on
the table.

## mlsys26 cross-check (`Ascend910BMixed::compute_cost`)

The model charges, per mixed group, the **symmetric cross-term** (fill folded inside the `max`):
`max(cube_stage + one_vec_tile, vec_stage + one_cube_tile, ddr_lat)` for a 2-stage shape,
`max(cube_stage, vec_stage, ddr_lat)` for a 3-stage one (fill absorbed), plus a per-launch
`rounds * kernel_fill_cost`. (The earlier additive `fill + max` is superseded; the measurements
below fit the cross-term — see README's model summary.)

| model assumption | verdict | evidence |
| --- | --- | --- |
| the `max(cube, vec, …)` **overlap** | **real, but only when skewed** | overlap `t/pipe → 0.95`; serial is `cube+vec`, **over-credited up to 2.26×** |
| an implicit **serial fallback = `cube+vec`** | **under-estimates the true serial** | serial `t/srl` 1.00→1.34 — real serialization also costs the intra-unit pipeline |
| a **`fill`** term | **= the bottleneck unit's INITIAL IDLE** (not "one cube tile") | Grounded across all 4 shapes (`c→v`, `v→c`, `v→c→v`, `c→v→c`): fill **adds** one producer-tile when the output stage's unit is idle at the start (2-stage — `c→v` fill=`cube_tile`, `v→c` fill=`vec1_tile`; both `t/(max+fill)=1.00`), and is **absorbed** when that unit already runs an earlier stage (3-stage `v→c→v`/`c→v→c` — `total==max` to ~1 cy). Amortizes 58%→18% (NT 1→8) when it adds |
| the **1:2 mix-cluster** (`cores_used=3·eff_units`) | **consistent** | `VEC_CORES_PER_AIC=2`; per-core AIV wall = half the vector work |
| the **`ddr_lat`** term | **grounded — single-core subsumed; cross-unit pool is the real term** | `mixed_ddr_bound` (K 16→512): the single-core GM is **subsumed** into the stages (`max(cube,vec,ddr) == max(cube,vec)`; the max GM port ≤ its stage; `ddr_cycles` fixed to max-over-ports). `mixed_contention` (multi-core): the cube (`GM_TO_L1`) + vector (`GM_TO_UB`) reads share **one** 900 GiB/s pool — past the knee both collapse to `900/B`, matching `par(active, peak)=min(peak, 900/B)` to **0%**. So `ddr_lat` is the *cross-unit shared-HBM* term and **only** that — a single-core ddr term double-counts |

**Net for the scheduler:** the model is right that a *skewed* mixed kernel overlaps to
`max(cube, vec)`, and (a) it folds the **fill** inside that `max` as the cross-term (one
non-bottleneck tile onto each stage) = the **bottleneck unit's initial idle**: adds for a
2-stage kernel, ~0 for a 3-stage kernel where the output unit does double-duty — the shipped
form, superseding the additive `fill + max`; (b) it must not credit the overlap for a **genuine
cross-tile carry / multi round-trip** (single round-trip — `c→v`, `v→c`, `v→c→v`, `c→v→c` — all
overlap; the serial fallback `cube+vec` even *under*-estimates the true serial by 1.0→1.34×);
and (c) it needs **no separate single-core `ddr` term** — subsumed into `max(cube, vec)`
(`mixed_ddr_bound`); `ddr_lat` earns its place only as the cross-unit shared-HBM-read contention.

## Status of the planned experiments

- **mixed_ddr_bound** — **done**. K 16→512 grounds the `ddr` term (subsumed into the stages,
  above). En route it also flips the bottleneck **STAGE** vec→cube at K≈128 (MAD grows with K;
  the `C=C+C` vector stage is K-independent GM load+store) and the AIC **dominant pipe**
  fixp(store)→mte2_aic(reload) — so the K-sweep already validates `max()` picks the right stage
  on the cube-growing side. Honest caveat: the AIC stays GM-bound throughout because each tile
  reloads `B[K,N]`, so reload ∝ K tracks MAD ∝ K; a true MAD-bound regime needs a B-resident
  kernel (load B once, reuse across tiles).
- **mixed_balance** *(optional)* — the K-sweep covers the cube-growing direction; a complementary
  epilogue-op-count sweep would confirm the vec-growing direction. Lower priority now.
- **mixed_contention** — **done**. Multi-core skewed kernel; the cube (`GM_TO_L1`) + vector
  (`GM_TO_UB`) reads share **one** 900 GiB/s pool — past the knee both collapse to `900/B`,
  matching `par(active, peak)=min(peak, 900/B)` to 0%. The cube read caps at B≈7 cores
  (900/135), the vector at B≈9 (900/100.9) — different knees, *same* pool. This is the only
  place a separate `ddr_lat` earns its keep. (Nuance: the perf-sim applies the pool as a
  per-pipe-per-core cap `min(peak, 900/B)`, not a summed byte-volume — but the pooling is real:
  one `total_read_gibs` knob throttles both read pipe-types. The cap is set in-kernel via
  `SetHillBandwidthModel`, which overrides `PTO_BW_MODE=fitted`, so the default binary's
  uncapped-vs-capped pair *is* the flat-vs-contention comparison.)
- **Shape sweep (mixed_vcv / mixed_vc / mixed_cvc)** — **done**. The 4 canonical
  single-round-trip shapes — `c→v` (epilogue), `v→c` (prologue), `v→c→v`, `c→v→c` (flash-decode)
  — grounding the fill rule in the cross-check table. `c→v→c` also confirms the two cube matmuls
  **overlap** (`t/srl → 0.49`, ~2× vs serial) under *separate* ping-pong buffers, validating
  upstream **#1900**'s per-stage buffer separation (its depth-2 skew is what unblocks this). The
  `fill` rule (bottleneck initial idle: adds for 2-stage, absorbed for 3-stage) holds in every
  shape's NTILES sweep, so `mixed_filldrain` is subsumed.
