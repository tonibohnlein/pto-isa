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
| chained matmul: intermediate C **excluded** from GM reload | `gml1_chain` | per-term error ≤0.1%; C round-trip = mm1 store + mm2 C-reload; fusion saves 23→30% as Ki grows |
| truly fused lowering (C resident in L1) hits the fused number | `gml1_fused` | `fused mte2 = reload(A,B,D)` to −0.0% (1–4 M-bands); C never TLOAD'd from GM; 40–44% reload saving |
| multi-core `par(active,peak) = min(active, HBM/peak)` | `gml1_multicore` | uncapped `mte2·B` constant (linear); capped per-core bw = `min(135,900/B)` ≤0.4%; aggregate saturates at HBM past the knee |

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

### Chained matmul: the intermediate never hits DDR (`gml1_chain`)

`cube_operand_reload()` walks the matmul subgraph and charges GM reload only for
**boundary** operands (`!produced.count(operand)`); an intermediate `C` produced by
`MM1` and consumed by `MM2` is on-chip ephemeral and excluded. With the single-matmul
`RunGemmE2E` we can't keep `C` resident, so we measure `MM1` (`A·B→C`) and `MM2`
(`C·D→E`) separately and decompose the GM traffic. The **C round-trip** that fusion
eliminates is exactly `MM1`'s C-store (`fixp`) + `MM2`'s C-reload (the lhs half of its
`mte2`); each term matches the model to ≤0.1%. Sweeping the shared dim `Ki = N1 = K2`,
the round-trip scales `~ M·Ki` (saving 23→30% of total GM traffic) while the boundary
reloads `A,B,D` are unchanged — the cost-model accounting.

`gml1_fused` then closes the loop with a **truly fused** single-core kernel
(`chain_fused_kernel.cpp`, `RunGemmChainFused`): MM1 computes `C` in L0C, `TMOV` drains
it L0C→L1 (`copy_matrix_cc_to_cbuf`, fp32→bf16), and MM2 `TEXTRACT`s `C` from L1 — `C`
never touches GM. Measured `fused mte2 = reload(A,B,D)` to **−0.0%** (zero C reload),
and 40–44% reload saving vs the unfused pair. So the produced-exclusion is not just an
accounting identity — a realizable lowering hits exactly the number the model scores.

## Multi-core (the `par` cap) [M] (`gml1_multicore`)

A single core streams at `BW_GM_L1 = 135 GiB/s`. With `n` active cores the mlsys26
model divides the per-core feed but caps the aggregate at HBM:

```
par(active, peak) = min(active, hbm_aggregate_gibps / peak)
```

The perf-sim's Hill model encodes **exactly** this: `BwEff(key, bytes, ncores) =
min(HillBw(bytes), total_read_gibs / ncores)`, and `LAUNCH_KERNEL` sets `ncores` from
the launch `block_dim`. Running `gemm_performance` partitioned along N across
`B ∈ {1,2,4,8,16}` cores, with the per-fid Hill model set to `total_read_gibs = HBM`:

| mode | per-core MTE2 | meaning |
| --- | --- | --- |
| **uncapped** (`total_read=0`) | `mte2·B` **constant** (856064) → `~1/B` | linear scaling, no contention = `par = active` |
| **capped** (`total_read=900`) | per-core bw = `min(135, 900/B)` to ≤0.4% | aggregate saturates at **900 GiB/s** for `B>6.7` |

So the perf-sim default (`total_read=0`) reproduces `par = active` — mlsys26's
`hbm_aggregate_gibps = 24·135 = 3240` (cap effectively disabled). Enabling the realistic
A3 cap (~900 GB/s) reproduces `par()` saturation: per-core MTE2 stops shrinking past the
`≈6.7`-core knee (B=8→16: 128000→128256, flat) because HBM is saturated. This is the
direct multi-core validation of the mlsys26 `par()` formula — splitting a reload-bound
matmul across more cores past the knee buys nothing.

## Flat vs fitted GM→L1 bandwidth [M] (`run.py --fitted`)

The perf-sim default (and our mlsys26 model) use the **flat** GM→L1 = 135 GiB/s. The
env-gated **fitted** Hill model (`PTO_BW_MODE=fitted`, `MakeFittedHillModel`) uses
`HillBw(B) = 28.61·B/(1107+B)` — saturating at **28.61 GiB/s** (~4.7× below flat),
"fixing GM_TO_L1's 0.70× low bias of the mixed fit" (arch_config.hpp). **The perf-sim
DOES consult it** — re-running the reload sweep under `PTO_BW_MODE=fitted` makes MTE2
**4.9–7.3× slower** (mean 5.4×), and the per-TLOAD Hill prediction `Σ B/HillBw(B)`
matches to **0.1%**. Two findings:

- **Bigger than the peak ratio.** 4.7× is the *peak* ratio; the measured 5.4× mean
  (up to 7.3× for 16×16 tiles) is larger because the Hill `k=1107 B` floor penalises
  **small** transfers: effective GM→L1 rises from 18.7 GiB/s (2 KiB TLOAD) toward the
  28.61 peak (32 KiB TLOAD).
- **Granularity-dependent.** Under flat, reload cost = `total_bytes/135` (only total
  matters). Under fitted it depends on the **TLOAD size** (tile dims, `stepK`) — a
  per-transfer effect the flat `bytes/bw` term *cannot* express.

Implication for mlsys26: `bw_gm_l1 = 135` is ~5× optimistic versus the on-device fit,
and a single-number calibration understates fine-tile reload. If reload accuracy
matters, adopt the Hill form (`peak=28.61, k=1107`) keyed on per-TLOAD bytes rather
than re-fitting one flat constant. (The fitted Hill curve is also where the multi-core
`par()` aggregate would bind — a future multi-core experiment.)

## Caveats

- Output-stationary tiles. Coverage: split-K **sink** per-worker (`gml1_splitk`);
  chained-matmul reload accounting by decomposition (`gml1_chain`) **and** a truly fused
  multi-band lowering (`gml1_fused`, C resident in L1, `M/bm` bands); the multi-core
  `par()` aggregate cap (`gml1_multicore`). Not yet exercised: split-K's `S·store`
  **write**-side HBM contention (the `par()` validation here is on the GM **read** group),
  and non-square / mixed-dtype operand combinations.
- The flat-vs-fitted gap is measured indirectly (the default build is flat); a
  `PTO_BW_MODE=fitted` sweep would measure the fitted curve directly.
- `bytes_a = bytes_b = 2` (bf16) throughout; fp32 operands (`cpr=2`, `kt=8`) change
  the compute pipe but not the reload structure.
