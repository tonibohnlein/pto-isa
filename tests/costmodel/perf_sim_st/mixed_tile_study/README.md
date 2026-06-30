# Mixed-kernel cost-model study

A perf-sim study that grounds the analytic **mixed cube+vector** cost model behind PyPTO's
scheduler — how a matmul (cube/AIC) feeding a pointwise/reduction epilogue (vector/AIV)
through a GM round-trip overlaps when software-pipelined. It is the **sibling** of
`gm_l1_tile_study` / `l0_tile_study` (cube reload/extract) and `vec_tile_study` (vector
compute): those scope a single unit, this one scopes the **cube↔vector pipeline**.

Everything builds under `-D__COSTMODEL` and runs the **host** pipeline simulator — no NPU
required. One command reproduces it:

```bash
python run.py             # generate -> build -> run -> analyze, all experiments
python run.py --no-build  # re-analyze existing CSVs
```

## Scope: the mixed cube+vector kernel (AIC↔AIV through GM)

On the 910b there is **no direct UB↔Mat/L1 path** — the cube (AIC) and vector (AIV) are
separate core pools that hand off only through **GM/DDR**. A "mixed" kernel is therefore a
cube matmul whose result the vector epilogue reads back from GM, software-pipelined so the
two units overlap (the cube computes tile `k+1` while the vector consumes tile `k`). This is
exactly what PyPTO's `ExpandMixedKernel` → `InjectGMPipeBuffer` → `SkewCrossCorePipeline`
passes build: split into AIC + AIV functions, a GM-backed FIFO between them, and a
producer-skew so they overlap.

The relevant perf-sim pipes (CSV columns, per unit row):

- **AIC**: `mte2_aic` (GM→L1), `mte1` (L1→L0), `cube` (MAD), `fixp` (L0C→GM — the handoff WRITE)
- **AIV**: `mte2_aiv` (GM→UB — the handoff READ), `vec` (compute), `mte3` (UB→GM)
- **`total_cycles`**: the kernel wall-clock. All pipes run on **one event clock**, so this is
  identical on every row; you read latency straight from it.

## The mechanism the perf-sim exposes

The perf-sim overlaps the AIC and AIV timelines **automatically** — pipes run concurrently
unless a data dependency links them (the `TileDepTracker` derives a cross-pipe wait by
matching the GM buffer address; **no FFTS / explicit flags** are needed for the cube→vector
handoff). So whether the kernel wall-clock looks like `max(cube, vec)` or `cube + vec` is
decided **purely by the data dependencies the kernel encodes**:

- **vector reads the cube's output buffer** (a per-tile RAW edge) → the two units **serialize**.
- **cube runs a tile ahead on a ping-pong buffer** → cube(k+1) has no dep on vector(k) → they
  **overlap**. This is the `SkewCrossCorePipeline` producer-skew.

> Caveat the study relies on: the dep tracker models **RAW only** (not WAR/WAW). So overlap
> needs distinct producer/consumer buffers + a one-tile skew; a true serial needs a real RAW
> edge from the vector's output back into the *next* cube's input.

## The cost model we ground against (mlsys26)

`Ascend910BMixed::compute_cost` charges, for a mixed group:

```
latency = fill + max(cube_stage, vec_stage, ddr_lat)        # full overlap, UNCONDITIONALLY
cores_used = 3 * eff_units                                   # the 1 cube : 2 vector mix-cluster
```

This study validates two things the model assumes: **(a)** the `max(...)` overlap is real
*only when the kernel is skewed* (else it degrades to the sum — or worse), and **(b)** the
`fill` term on short tile loops.

## Experiments & findings

Two kernels over the **same** tiled `matmul → pointwise` work (NTILES output tiles along M),
sweeping `NTILES ∈ {1,2,4,8}`:

| experiment | encoding | result |
| --- | --- | --- |
| **mixed_overlap** | skewed ping-pong: cube(k+1) ∥ vector(k) on alternating GM buffers | `total = max(cube, vec) + fill`; **fill = exactly one cube tile** (amortizes 58%→18%); `overlap_factor` 0→0.84 |
| **mixed_serial** | per-tile RAW chain: cube(k) takes its B-operand from the handoff buffer the prior vector wrote, so it waits vector(k−1) | `total = the sum, and EXCEEDS it` (1.0→1.34×) — isolating each tile also kills intra-AIC cross-tile pipelining |
| **mixed_ddr_bound** | the skewed kernel, sweeping K 16→512 (cube MAD grows; the `C=C+C` GM stages don't) | `total = max(cube, vec) + fill` holds across the *whole* compute↔GM-bound sweep; the **`ddr` is subsumed** into the stages (`max(cube,vec,ddr) == max(cube,vec)`), never a separate term. Bottleneck stage flips vec→cube at K≈128 |

Measured (bm=128, N=128, K=128, fp16 in / fp32 acc, `C = C + C` epilogue):

| NTILES | overlap `total` | serial `total` | speedup | fill share | overlap_factor |
| ------ | --------------- | -------------- | ------- | ---------- | -------------- |
| 1 | 4188 | 4188 | 1.00× | 58% | 0.00 |
| 2 | 5489 | 7925 | 1.44× | 44% | 0.43 |
| 4 | 8139 | 15399 | 1.89× | 30% | 0.70 |
| 8 | 13439 | 30347 | **2.26×** | 18% | 0.84 |

- **Overlap → `max(cube, vec) + fill`**: `total/(max+fill)` rises 0.86→0.95 (slightly under 1.0
  because `vec < cube`, so the drain tail is shorter than a full fill). `fill = AIV active_start`
  = one cube tile (2426 cy @ bm128), constant in NTILES.
- **Serial → the sum, and beyond it**: `total/(cube+vec)` rises 1.00→1.34. Forcing every tile
  behind the handoff also serializes the cube's *own* cross-tile pipeline — the per-tile cube
  wall is the isolated 2426 cy vs the pipelined 1463 cy in the overlap run. So the no-skew
  penalty is **worse** than a naive `cube + vec` would predict.
- **Speedup serial→overlap**: 1.0× (NT1) → 2.26× (NT8), approaching `(cube + vec) / max`.

## How mlsys26 diverges (the grounding gaps this study targets)

1. **Overlap is credited unconditionally.** `max(cube, vec, ddr)` is the *skewed* cost. The
   `SkewCrossCorePipeline` **demote-to-sequential** path (consumer-role and multi-round-trip
   loops) produces the **serial** kernel — which the model over-credits by up to **2.26×**.
   And the model's implicit serial fallback (`cube + vec`) itself **under-estimates** the true
   serial by 1.0→1.34×, because real serialization also costs the intra-unit pipeline.
2. **No fill/drain term in practice.** The producer-skew prologue is **exactly one cube tile**
   — 58% of the wall at NTILES=1, still 18% at NTILES=8. Single-/few-tile mixed kernels pay
   it in full; the model's `fill` must carry it.
3. **The 1:2 mix-cluster holds** (`VEC_CORES_PER_AIC = 2`), but the per-core AIV wall is half
   the total vector work — the `/(2·eff_units)` division is consistent with the sim.

## Files

- `common.py` — a2a3 constants, the analytic stage model (`cube_stage_cycles` /
  `vec_stage_cycles` / `ddr_cycles`, `predict_serial` / `predict_pipelined` /
  `mlsys_mixed_latency`), the perf-sim CSV reader `read_mixed` (wall-clock + per-unit windows)
  and `overlap_factor`, and the self-contained testcase emitter.
- `experiments.py` — the `mixed_overlap` (skewed ping-pong) and `mixed_serial` (RAW-chain)
  generators + the inline kernels (cube tile = `RunGemmE2E`, vector tile = in-place
  TLOAD/op/TSTORE).
- `analyze.py` — pairs the two runs per (shape, NTILES) and grounds against the predictions.
- `run.py` — generate → build → run → analyze.
- `DESIGN_SPACE.md` — the mixed design space and the mlsys26 model cross-check.
