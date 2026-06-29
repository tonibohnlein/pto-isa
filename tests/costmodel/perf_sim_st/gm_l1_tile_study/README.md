# GM→L1-tile cost-model study

A perf-sim study that validates the analytic **GM↔L1** cost model behind PyPTO's
GM→L1 autotiler — the reload / output-store terms our mlsys26 Ascend-910B cube
cost model scores (`Ascend910BCost::cube_operand_reload` + the cube DDR roofline).
It is the **sibling** of `l0_tile_study/`: that study scopes the L1→L0 boundary
(compute-bound); this one scopes the GM↔L1 boundary (memory-bound).

Everything builds under `-D__COSTMODEL` and runs the **host** pipeline simulator —
no NPU required. One command reproduces all of it:

```bash
python run.py            # generate -> build -> run -> analyze, all experiments
python run.py gml1_reload # a single experiment
python run.py --no-build  # re-analyze existing CSVs
python run.py gml1_reload --fitted  # also run PTO_BW_MODE=fitted + Hill analysis
```

## Scope: GM ↔ L1 only

The level **above** the L0 chooser stages operands GM→L1 and drains the matmul
output L0C→GM. That boundary is where "GEMM is memory-bound" actually lives — the
operands are streamed from HBM with low arithmetic intensity and reloaded once per
output-tile block. The `l0_tile_study` deliberately treats `GM→L1` (MTE2) as
out-of-scope noise; **this study makes it the object of study.**

We drive the same `gemm_performance` reference kernel (`RunGemmE2E`) on a **single
core** (`blockDim=1`, `singleCore = whole problem`) so the AIC pipeline summary
measures exactly the per-core reload our model scores. Its `TLOAD`s (GM→L1, into a
`MatTile`) are the operand reload; its `TSTORE`s (L0C→GM) are the FixPipe drain.

## a2a3 cost model

From `include/pto/costmodel/arch_config.hpp` (`BandwidthTable`, flat/legacy a2a3):

| quantity | value |
| --- | --- |
| **GM → L1** (MTE2 reload port) | **135 GB/s** — the term our mlsys26 model calls `bw_gm_l1` |
| L0C → GM (FIXPIPE output store) | 70 GB/s |
| L1 → L0A / L1 → L0B | 441 / 220.5 GB/s |
| frequency | 1.85 GHz |
| MAD (one TMATMUL) | `6 + cpr·⌈m/16⌉·⌈k/kt⌉·⌈n/16⌉`, `kt=32/bytes_a`, `cpr=2 fp32 / 1 bf16` |
| transfer cycles | `bytes / 2³⁰ / bw[GiB/s] · freq` |

A TLOAD into a `MatTile` resolves to `PipeKey::GM_TO_L1` and is charged at the
**flat** table value (135) by default. Under `PTO_BW_MODE=fitted` (`run.py --fitted`)
the perf-sim instead uses the on-device Hill fit `28.61·B/(1107+B)`, which makes MTE2
**4.9–7.3× slower** (mean 5.4×, Hill per-TLOAD model exact to 0.1%) — bigger than the
4.7× peak ratio because the `k=1107 B` floor penalises small TLOADs. Our mlsys26 model
hardcodes the flat 135; the gap and its granularity-dependence are in `DESIGN_SPACE.md`.

## The model term validated here

Our mlsys26 `cube_operand_reload`, for one matmul `C[M,N] += A[M,K]·B[K,N]` with an
output tile `(h=baseM, w=baseN)`:

```
reload = M·N·K / w · bytes_a    # A panel reloaded once per N-block (N/w times)
       + M·N·K / h · bytes_b    # B panel reloaded once per M-block (M/h times)
```

This is **exactly** the GM→L1 byte volume `RunGemmE2E` issues: A is `TLOAD`ed for
every `(i,j,kIter)` (no cross-`j` residency → reloaded `N/baseN` times), B likewise
reloaded `M/baseM` times. The K-staging knobs `stepKa/stepKb` only batch the TLOADs
into bigger L1 panels — they do **not** change the total bytes.

## Experiments & findings

| experiment | validates | result |
| --- | --- | --- |
| **gml1_reload** | reload byte formula + GM→L1 bandwidth (sweep `(baseM,baseN)`) | MTE2 cycles match `reload/135` to **0.3% mean error** over 24 tiles (3→32 MiB); effective BW **135.4 GiB/s**, spread 0.5% |
| **gml1_roofline** | `total == max(mte2,mte1,cube,fixp)` (overlap, not sum) across regimes | `t/max ≈ 1.00–1.06`, `t/sum ≈ 0.45` for 4/5 regimes → pipes overlap, feed/drain are separate |
| **gml1_stepk** | MTE2 invariant to K-staging depth | mte2 **exactly** flat across `stepK∈{1,2,4}` → model correctly omits a stepK term |
| **gml1_splitk** | split-K sink: feed/compute `~ Kc=K/S`, output store a constant floor | `mte2/Kc` flat (104.5), `cube ∝ Kc`, `fixp` 0.0% spread; bound flips MTE2→FIXP at the knee — validates `eval_S` |
| **gml1_chain** | chained `C=A·B`, `E=C·D`: intermediate C excluded from GM reload | per-term error **≤0.1%**; the C round-trip (mm1 store + mm2 C-reload) is exactly what fusion drops — **23→30% saving** as Ki grows |

### The one regime where `max` is optimistic

`skinny_membound` (1024×1024×128, large output / tiny K) gives `t/max = 1.30`:
here `mte2 ≈ fixp` (both ≈53k cycles) and the **single-buffered L0C** (`dbC=1` in
`RunGemmE2E`) exposes the output drain — the store cannot hide under the tiny
2-iteration K-loop. The GM-read feed and GM-write drain then partially serialize.
This is the same drain-exposure the `l0_tile_study` `dbc` experiment found
(`depthC=2` wins 13–37% when drain-bound), and it marks the regime where our
`ddrS = max(feed, writes)` simplification is most optimistic. See `DESIGN_SPACE.md`.

### The output store drains as bf16

`gml1_splitk` pins the FixPipe store cost: `fixp = M·N·2 / BW_L0C_GM` to +0.1%
(512² and 1024²). The FixPipe drains the **fp32** L0C accumulator to GM as a
**2-byte (bf16)** write at 70 GiB/s — so `out_store`'s width is the **output (drain)
dtype**, not the 4-byte accumulator. mlsys26's `out_store` uses
`dtype_bytes(output tensor)`, which is correct iff that output is bf16; an
fp32-output matmul would be charged 2× the sim's store floor.

## Files

- `common.py` — a2a3 constants, the `reload_bytes`/`transfer_cycles`/`mad_cycles`
  model, the perf-sim CSV reader, and the `RunGemmE2E` single-core emitter.
- `experiments.py` — testcase generators (also listed in `../testcase/CMakeLists.txt`).
- `analyze.py` — reads the CSVs, scores each prediction.
- `run.py` — generate → build → run → analyze.
- `DESIGN_SPACE.md` — the GM↔L1 design space and the mlsys26 model cross-check.
