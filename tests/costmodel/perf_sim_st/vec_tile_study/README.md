# Vector-tile cost-model study

A perf-sim study that grounds and validates the analytic **vector-core** cost model
behind PyPTO's scheduler — the per-op compute formula, the GM↔UB roofline, and the
vector parallelism/streaming mechanisms our mlsys26 model scores. It is the **sibling**
of `gm_l1_tile_study` (GM↔L1 cube reload) and `l0_tile_study` (L1→L0 cube extract):
those scope the **cube** cores, this one scopes the **vector** cores.

Everything builds under `-D__COSTMODEL` and runs the **host** pipeline simulator — no
NPU required. One command reproduces it:

```bash
python run.py             # generate -> build -> run -> analyze, all experiments
python run.py vec_pointwise   # a single experiment
python run.py --no-build  # re-analyze existing CSVs
```

## Scope: the vector cores (VEC / GM↔UB)

The vector cores run **pointwise** and **reduction** ops on UB-resident tiles, streamed
GM↔UB. The relevant pipes (perf-sim CSV columns, `unit=AIV0/AIV1`):

- **`vec_cycles`** (`PipeKey::VECTOR`) — the compute: `vadd/vmul/vexp/vreducev2/...`
- **`mte2_aiv_cycles`** (`GM→UB`) — the vector load (validated as a shared HBM read pool
  in `gm_l1_tile_study/gml1_contention`)
- **`mte3_cycles`** (`UB→GM`) — the vector store

## The cost model we ground against (pto-isa stub)

The perf-sim uses the **stub** backend (`cce_costmodel/`), device-calibrated per
instruction (910B3 标定, R²≈1.0). The vector compute formula
(`cce_costmodel_core.hpp:133`, `EstimateLinearCycles`):

```
cycles = slope · repeat                              # per SIMD repeat (256-B reg: 64 fp32 / 128 fp16)
       + head + tail   IF the vec pipe queue is empty (stream start — paid ONCE)
       + 16            (binary ALU, when cols are not repeat-aligned: count-mode floor)
```

Back-to-back vector ops in one stream **overlap their startup latency** — `head+tail` is
paid once, not per op. `repeat = ROWS·COLS / (256/sizeof(T))` for a VL-aligned tile.

Device-calibrated coefficients (`cce_costmodel_vector_compute.hpp`), as `(slope, head+tail)`:

| op (lowers to) | slope | head+tail |
| --- | --- | --- |
| `vadd/vsub/vmax/vmin` | 2 | 24 |
| `vmul` | 2 | 25 |
| `vexp` / `vln` | 2 | 31 / 33 |
| `vdiv` | 4 | 30 |
| `vrelu/vmuls/vmaxs/vrsqrt` | 1 | 23–26 |
| `vreducev2` (reduction) | 14 | 34 |

## Experiments & findings

| experiment | grounds | result |
| --- | --- | --- |
| **vec_pointwise** | per-op `slope·repeat + once-per-stream (head+tail)` | every op matches the stub to **0.0%** (add 2/24, mul 2/25, div 4/30, exp 2/31); the per-op startup is paid **once per chain**, so mlsys26's per-op charge overcounts fused chains **1.2×→2.7×** (NOPS 1→16) |
| **vec_reduce** | reductions are barrier-separated **trees**, not a single `slope·repeat` op | `TROWSUM` (reduce W) = `45·(COLS/64)+6` — **ROWS-independent**; `TCOLSUM` (reduce H) = `16(R-1)+30·log₂R` — both to **0.0%**. mlsys26's `repeat=ROWS·COLS/64` overcounts a tall-tile `TROWSUM` up to **19×** |
| **vec_stream** | UB-overflow streaming recompute factor (`N_passes`) | online-streamed softmax wide body is **flat (≤1×)** across NCHUNKS 1→8 (`exp` runs once per element); mlsys26's `#reductions+1 = 3×` multiplier is **3–4× pessimistic** on every streamed softmax |

## How mlsys26 diverges (the grounding gaps this study targets)

mlsys26's vector model (`Ascend910BCost`, `set_910b`) uses `slope_pw=2, slope_reduce=14,
head=14, tail=18`, charged **per op**:

1. **Startup is per-op, not per-stream** — a fused softmax (exp+reduce+div, one stream) is
   charged `3×32` fixed cycles vs the device's `~34` once. Overcounts fused chains (the
   `vec_pointwise` chain sweep measures 1.2×→2.7×).
2. **Single pointwise slope** — collapses the real spread (`vmuls/vrelu`=1, `vdiv`=4,
   `vexp` higher startup) into one slope=2.
3. **No count-mode floor** — the +16 unaligned-width penalty is unmodeled.

## Files

- `common.py` — a2a3 constants, the device-calibrated `EstimateLinearCycles` model + per-op
  coefficients, the perf-sim CSV reader (`read_aiv`), and the self-contained kernel emitter.
- `experiments.py` — testcase generators + the inline `VecChain` kernel (also listed in
  `../testcase/CMakeLists.txt`).
- `analyze.py` — reads the CSVs, grounds each prediction against the stub.
- `run.py` — generate → build → run → analyze.
- `DESIGN_SPACE.md` — the vector design space and the mlsys26 model cross-check.
