# Single-core GM↔L1 GEMM — the design space (memory-bound boundary)

Scope: **one core, operands streamed from GM/HBM into L1, output drained L0C→GM.**
`C[m,n] += A[m,k]·B[k,n]`. This is the tiling the level **above** the L0 chooser
solves — the memory-bound boundary our mlsys26 Ascend-910B cube cost model scores.
Multi-core enters only through a per-core bandwidth divide (see "Multi-core").

Confidence tags: **[V]** verbatim in a primary source · **[D]** derived · **[M]**
measured here against the perf-sim.

## The 3 orthogonal axes

| # | Axis | the choice | what it costs |
| --- | --- | --- | --- |
| **1** | **Output tile `(h=baseM, w=baseN)`** | how big a C tile each core pins in L0C | sets operand **reload** (`N/w` reloads of A, `M/h` of B) |
| **2** | **K-staging depth `stepKa/stepKb`** | how much of the K-panel is staged into L1 per TLOAD | sets L1 footprint + TLOAD batching; **not** total bytes |
| **3** | **L0C drain buffering `dbC`** | single vs double-buffered output accumulator | whether the FixPipe (L0C→GM) drain hides under compute |

Axis 1 is the dominant memory-traffic dial; axes 2–3 are overlap dials that move
`total` without moving the reload byte volume.

## The cost model (what we score)

For a single matmul, one core, output tile `(h, w)`:

```
reload  = M·N·K/w · bytes_a  +  M·N·K/h · bytes_b      # GM→L1 (MTE2), the dominant term
store   = M·N · bytes_c   (bytes_c = OUTPUT dtype, 2 B) # L0C→GM (FixPipe drain), shape-only
feed    = reload / BW_GM_L1                             # cycles
writes  = store  / BW_L0C_GM                            # cycles
ddr     = max(feed, writes)        # GM-read and GM-write are SEPARATE concurrent pipes
compute = cube_cycles(M,N,K,h,bk,w)                     # the 4th roofline pipe (MTE1, CUBE)
wall    = max(ddr, compute)        # roofline; double-buffering makes max (not sum) hold
```

This mirrors `Ascend910BCost`: `cube_operand_reload()` = `reload`; the cube DDR
`eval_S` lambda computes `feed`/`writes`/`ddrS = max(feed, writes)`; `db_roofline`
takes the `max` over the DDR and compute pipes.

## What the perf-sim confirms (measured) [M]

| model claim | experiment | verdict |
| --- | --- | --- |
| `reload = MNK/w·ba + MNK/h·bb` is the GM→L1 byte volume | `gml1_reload` | **0.3% mean error** over 24 tiles (3→32 MiB) |
| GM→L1 charged at flat **135 GiB/s** | `gml1_reload` | effective BW **135.4 GiB/s**, spread 0.5% |
| `wall = max(pipes)` (overlap, not sum) | `gml1_roofline` | `t/max≈1.0`, `t/sum≈0.45` (4/5 regimes) |
| feed (GM→L1) ∥ drain (L0C→GM) → `max`, not `+` | `gml1_roofline` | confirmed by `t/sum≈0.5` |
| reload bytes independent of `stepK` | `gml1_stepk` | mte2 **exactly** flat across stepK∈{1,2,4} |
| split-K: feed/compute `~ Kc`, store a constant floor | `gml1_splitk` | `mte2/Kc` flat (104.5), `cube ∝ Kc`, `fixp` 0.0% spread; bound flips MTE2→FIXP at the knee |
| output store width = **output dtype (2 B)**, not 4-B accumulator | `gml1_splitk` | `fixp = M·N·2/70` to +0.1% (512² and 1024²) |

### Where `max(feed, writes)` is optimistic [M]

`skinny_membound` (1024×1024×128, large output / tiny K): `mte2≈fixp` both saturate,
`dbC=1` exposes the drain (the 2-iteration K-loop can't hide a 2 MiB bf16 store), and the
GM-read/GM-write pipes partially serialize → `total ≈ 1.3·max`. The `max(feed,
writes)` form assumes the drain fully overlaps the feed; that holds whenever **either**
compute hides the drain (deep K) **or** `dbC=2`. In the exposed corner it is
optimistic by ~30%. This is the GM↔L1 image of the `l0_tile_study` `dbc` result
(`depthC=2` wins 13–37% when drain-bound) — the same lowering knob (`dbC`) governs it.

### Split-K (sink): the store floor that bounds the split (`gml1_splitk`)

A parallel split-K **sink** launches `S` workers, each computing a full `M×N` partial
over a `K/S` slice and atomic-adding it to GM. Per worker: `feed`/`compute ∝ Kc=K/S`,
but the output `store` is a **constant** `M·N·2/BW_L0C_GM` floor (independent of `Kc`).
Measured: `mte2/Kc` flat at 104.5, `cube` halves with `Kc`, `fixp` constant (0.0%
spread). So the per-core wall shrinks with `S` only while feed-bound, then plateaus at
the store floor (bound flips MTE2→FIXP at the knee). This is exactly what the mlsys26
`eval_S` enumeration trades off — `ddrS = max(feed, S·writes)` with `writes` the
constant store. The **upward** re-inflation at large `S` (aggregate `S·store`
saturating HBM via `par()`) is multi-core and not single-core visible here.

The store-width subtlety: the FixPipe drains the **fp32** L0C accumulator to GM as a
**2-byte (bf16)** write at `BW_L0C_GM=70` (`fixp = M·N·2/70`, verified to +0.1% at
512² and 1024²). So `out_store`'s `bytes_c` is the **output (drain) dtype**, not the
4-byte accumulator. mlsys26 uses `dtype_bytes(output tensor)` — correct iff that output
is bf16; an fp32-output matmul would be charged 2× the sim's store floor.

## Multi-core (the `par` cap)

A single core streams at `BW_GM_L1 = 135 GiB/s`. With `n` active cores the mlsys26
model divides the per-core feed but caps the aggregate at HBM:

```
par(active, peak) = min(active, hbm_aggregate_gibps / peak)
```

Currently `hbm_aggregate_gibps = 24·135 = 3240` (effectively disabling the cap, so
`par = active`). The perf-sim AIC row is single-core, so this study does **not**
exercise the cap — but it pins the per-core peak (135) the cap divides. The realistic
aggregate HBM (~900 GB/s, pto-isa A3) would bind at `≈6.7` cores for a pure-reload
matmul; raising the cap was a deliberate choice (see the mlsys26 model notes).

## Flat vs fitted GM→L1 bandwidth [V]

The perf-sim default (and our mlsys26 model) use the **flat** GM→L1 = 135 GiB/s.
The env-gated **fitted** Hill model (`PTO_BW_MODE=fitted`,
`MakeFittedHillModel`) uses `HillBw(bytes) = 28.61·bytes/(1107+bytes)` — saturating
at **28.61 GiB/s**, ~4.7× below flat, "fixing GM_TO_L1's systematic 0.70× low bias
of the mixed fit" (arch_config.hpp comment). Implication for mlsys26: a reload-bound
matmul costed at the flat 135 is **optimistic by up to 4.7×** versus the on-device
fit. Whether to switch `bw_gm_l1` to the fitted value is a model-calibration
decision — out of scope here, but this study is the harness to settle it (rerun any
experiment under `PTO_BW_MODE=fitted` and re-fit `eff_GiB/s`).

## Caveats

- Single matmul, single core, output-stationary (`RunGemmE2E`'s only mode). Split-K
  **sink** reduction is exercised per-worker (`gml1_splitk`); chained matmuls and the
  multi-core aggregate (`S·store` HBM contention via `par()`) are not yet exercised.
- The flat-vs-fitted gap is measured indirectly (the default build is flat); a
  `PTO_BW_MODE=fitted` sweep would measure the fitted curve directly.
- `bytes_a = bytes_b = 2` (bf16) throughout; fp32 operands (`cpr=2`, `kt=8`) change
  the compute pipe but not the reload structure.
