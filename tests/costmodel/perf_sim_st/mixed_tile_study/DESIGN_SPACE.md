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

The model charges `latency = fill + max(cube_stage, vec_stage, ddr_lat)` for every mixed group.

| model assumption | verdict | evidence |
| --- | --- | --- |
| the `max(cube, vec, …)` **overlap** | **real, but only when skewed** | overlap `t/pipe → 0.95`; serial is `cube+vec`, **over-credited up to 2.26×** |
| an implicit **serial fallback = `cube+vec`** | **under-estimates the true serial** | serial `t/srl` 1.00→1.34 — real serialization also costs the intra-unit pipeline |
| a **`fill`** term | **needed; = one cube tile** | fill = AIV `active_start` = 2426 cy (bm128), 58% of the wall at NT=1, 18% at NT=8 |
| the **1:2 mix-cluster** (`cores_used=3·eff_units`) | **consistent** | `VEC_CORES_PER_AIC=2`; per-core AIV wall = half the vector work |
| the **`ddr_lat`** term | **grounded — single-core subsumed; cross-unit pool is the real term** | `mixed_ddr_bound` (K 16→512): the single-core GM is **subsumed** into the stages (`max(cube,vec,ddr) == max(cube,vec)`; the max GM port ≤ its stage; `ddr_cycles` fixed to max-over-ports). `mixed_contention` (multi-core): the cube (`GM_TO_L1`) + vector (`GM_TO_UB`) reads share **one** 900 GiB/s pool — past the knee both collapse to `900/B`, matching `par(active, peak)=min(peak, 900/B)` to **0%**. So `ddr_lat` is the *cross-unit shared-HBM* term and **only** that — a single-core ddr term double-counts |

**Net for the scheduler:** the model is right that a *skewed* mixed kernel overlaps to
`max(cube, vec)`, but it (a) must add the one-tile `fill`, (b) must not credit the overlap when
the loop can't be skewed (consumer-role / multi-round-trip → serial, which is *worse* than its
`cube+vec` fallback), and (c) needs **no separate single-core `ddr` term** — the GM traffic is
subsumed into `max(cube, vec)` (grounded by `mixed_ddr_bound`); `ddr_lat` earns its place only
as the cross-unit shared-HBM-read contention.

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
- **mixed_filldrain** — covered by the NTILES sweep (`fill` = one cube tile).
