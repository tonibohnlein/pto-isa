# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Shared constants, the analytic MIXED cube+vector cost model, and a perf-sim CSV reader for
# the MIXED-kernel cost-model study. See README.md for the full write-up.
#
# Sibling of gm_l1_tile_study / l0_tile_study (CUBE) and vec_tile_study (VECTOR): this study
# scopes the MIXED cube+vector kernel -- a matmul (AIC) feeding a pointwise/reduction epilogue
# (AIV) through a GM round-trip, software-pipelined so the two units overlap. It grounds the
# mlsys26 mixed model (Ascend910BMixed: latency = max(cube_stage, vec_stage, ddr)) against the
# perf-sim's event-driven AIC/AIV co-simulation.
#
# Probe finding (testcase/mixed_probe): the perf-sim runs ALL pipes on one event clock; the
# `total_cycles` column IS the kernel wall-clock (identical on every row). Whether it equals
# max(AIC, AIV) or AIC+AIV is decided purely by the data dependencies the kernel encodes:
#   - vector reads the cube's output buffer (per-tile dep)  -> serialized, total = AIC + AIV
#   - cube runs a tile ahead on a ping-pong buffer (skew)   -> overlapped, total = max(AIC, AIV)
# No FFTS / explicit flags needed for the cross-core handoff -- the GM data dep is sufficient
# (a reused ping-pong address needs a flag for the anti-dependency; see experiments.py).

import csv
import pathlib

# --- paths (self-contained; results land under results/) ---
STUDY_DIR = pathlib.Path(__file__).resolve().parent
PERF_SIM_ROOT = STUDY_DIR.parent                      # tests/costmodel/perf_sim_st
TESTCASE_DIR = PERF_SIM_ROOT / "testcase"
RESULTS_DIR = STUDY_DIR / "results"
CSV_DIR = RESULTS_DIR / "perf_sim_output"             # where the kernels write *_pipeline_summary.csv
CSV_DIR_FITTED = RESULTS_DIR / "fitted" / "perf_sim_output"  # PTO_BW_MODE=fitted run (run.py --fitted)
# The cube half reuses the proven RunGemmE2E kernel; testcases include its dir (see write_testcase).
GEMM_KERNEL_INCLUDE = "${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance"

# --- a2a3 constants (pto-isa include/pto/costmodel/arch_config.hpp) ---
FREQ_HZ = 1.85e9
BW_GM_L1 = 135.0      # GB/s  GM -> L1   (cube operand reload, MTE2_AIC)
BW_L1_L0A = 441.0     # GB/s  L1 -> L0A  (cube extract, MTE1; A/"left" port)
BW_L1_L0B = 220.5     # GB/s  L1 -> L0B  (B/"right" port; half of A)
BW_L0C_GM = 70.0      # GB/s  L0C -> GM  (FIXPIPE: the matmul output store == the handoff WRITE)
BW_GM_UB = 100.9      # GB/s  GM -> UB   (vector reload, MTE2_AIV == the handoff READ)
BW_UB_GM = 188.46     # GB/s  UB -> GM   (vector store, MTE3)
HBM_AGGREGATE_GIBS = 900.0   # shared HBM read-pool cap (validated in gm_l1 / vec studies)
VEC_CORES_PER_AIC = 2        # 1 AIC : 2 AIV mix-cluster (config.hpp:20); == mlsys26's 3*eff_units
VEC_REG_BYTES = 256          # vector register: 64 fp32 / 128 fp16 per SIMD repeat

BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

# --- mlsys26 mixed model (Ascend910BMixed::compute_cost) -----------------------------
# SHIPPED form (2-stage): latency = max(cube_stage + one_vec_tile, vec_stage + one_cube_tile, ddr)
#              (3-stage): latency = max(cube_stage, vec_stage, ddr)          -- fill absorbed
# The fill is folded INSIDE the max as the symmetric cross-term (each stage + one tile of the
# OTHER unit), so it ADDS for a 2-stage shape and is ABSORBED for a 3-stage / DDR-bound one --
# NOT the older additive `fill + max`. This study validates (a) the overlap is real only when
# the kernel is skewed (else it degrades to the sum), and (b) the fill rule; predict_pipelined
# (max + fill) below is the additive proxy the sim fits in the compute-bound sweep, equivalent
# to the cross-term there.
MLSYS_SLOPE_PW = 2.0
MLSYS_SLOPE_REDUCE = 14.0
MLSYS_HEAD = 14.0
MLSYS_TAIL = 18.0


def ceil_div(a, b):
    return (a + b - 1) // b


def transfer_cycles(byts, bw_gibs):
    """Perf-sim memory-pipe busy cycles for `byts` at `bw_gibs` GiB/s (flat model).
    Mirrors EstimateBandwidthCycles: bytes / 2**30 / bw * freq_hz.
    """
    return byts / (1024.0 ** 3) / bw_gibs * FREQ_HZ


# --- analytic mixed model: the AIC stage, the AIV stage, and the GM (ddr) bound ------
# Each stage is the per-UNIT wall time (the unit pipelines its own pipes internally). The two
# units then overlap (skewed) or serialize (per-tile dep). We read the actual per-unit windows
# from the CSV (read_mixed) and compare against these predictions.

def cube_stage_cycles(M, N, K, bm, bn, bk, bytes_a=2):
    """AIC per-unit wall time for a tiled [M,N]<-[M,K]@[K,N] matmul: the max of the cube's
    internal pipes (reload GM->L1, extract L1->L0, MAD, fixpipe L0C->GM), which double-buffer.
    First cut reuses the grounded gm_l1/l0 transfer model; refined against the CSV.
    """
    ntiles = (M // bm) * (N // bn)
    # reload: A reloaded N/bn times, B reloaded M/bm times (output-stationary)
    reload = transfer_cycles(M * N * K / bn * bytes_a + M * N * K / bm * bytes_a, BW_GM_L1)
    extract = transfer_cycles(M * N * K * bytes_a / bn, BW_L1_L0A) + \
        transfer_cycles(M * N * K * bytes_a / bm, BW_L1_L0B)
    store = transfer_cycles(ntiles * bm * bn * 4, BW_L0C_GM)   # fp32 accumulate out
    return max(reload, extract, store)


def vec_stage_cycles(M, N, nops, op_slope=2, op_ht=24, bytes_t=4):
    """AIV per-unit wall time for a pointwise epilogue over the [M,N] output (nops chained ops).
    The vector unit splits the work across AIV0/AIV1, so per-core work is halved. First cut:
    max(load GM->UB, compute, store UB->GM); refined against the CSV.
    """
    per_core = (M * N) // VEC_CORES_PER_AIC
    epr = VEC_REG_BYTES // bytes_t
    repeat = per_core // epr
    compute = op_ht + nops * op_slope * repeat
    load = transfer_cycles(per_core * bytes_t, BW_GM_UB)
    store = transfer_cycles(per_core * bytes_t, BW_UB_GM)
    return max(compute, load, store)


def ddr_cycles(M, N, K, bm, bn, bytes_a=2, bytes_t=4):
    """Single-core GM critical contribution = MAX over the four GM ports (NOT a sum):
      mte2_aic  cube reload GM->L1 : (A=M*K + B reloaded per tile=(M/bm)*K*N) * bytes_a / BW_GM_L1
      fixp      cube store L0C->GM : M*N * bytes_a / BW_L0C_GM  (perf-sim charges 2 bytes/elem here)
      mte2_aiv  vector reload GM->UB: M*N * bytes_t / BW_GM_UB  (full tile -- AIV is not split)
      mte3      vector store UB->GM : M*N * bytes_t / BW_UB_GM

    IMPORTANT: this single-core GM cost is SUBSUMED into cube_stage / vec_stage. The perf-sim's
    per-unit ACTIVE wall is the overlapped critical path through that unit's pipes (incl. these GM
    ports), so each port time is <= the stage that contains it -> max(cube,vec,ddr)=max(cube,vec).
    The OLD version SUMMED cube_store+vec_load+vec_store -- wrong twice: those ports overlap each
    other AND already sit inside the stages, so the sum over-read `ddr` past the real bottleneck.
    The mlsys26 `ddr_lat` term is only meaningful as the CROSS-UNIT shared-HBM-read contention
    (PTO_BW_MODE=fitted), which the flat model cannot express -- to be grounded separately.
    """
    ntiles = M // bm
    mte2_aic = transfer_cycles((M * K + ntiles * K * N) * bytes_a, BW_GM_L1)
    fixp = transfer_cycles(M * N * bytes_a, BW_L0C_GM)
    mte2_aiv = transfer_cycles(M * N * bytes_t, BW_GM_UB)
    mte3 = transfer_cycles(M * N * bytes_t, BW_UB_GM)
    return max(mte2_aic, fixp, mte2_aiv, mte3)


def predict_serial(cube_stage, vec_stage):
    """Per-tile-dependency kernel: no overlap. total ~= cube + vec."""
    return cube_stage + vec_stage


def predict_pipelined(cube_stage, vec_stage, fill):
    """Skewed (ping-pong) kernel: the two units overlap; total ~= max + one stage of fill/drain."""
    return max(cube_stage, vec_stage) + fill


def mlsys_mixed_latency(cube_stage, vec_stage, ddr):
    """mlsys26's prediction: max(cube, vec, ddr), UNCONDITIONALLY (no serial/skew distinction)."""
    return max(cube_stage, vec_stage, ddr)


# --- perf-sim CSV reader ------------------------------------------------------------
def read_mixed(fid, csv_dir=None):
    """Per-unit pipe windows for a mixed cube+vector kernel function `fid`.

    Returns:
      total      kernel wall-clock (the `total_cycles` column; identical on every row)
      aic        dict: active window + cube pipes (mte2_aic, mte1, cube, fixp)
      aiv        dict: per-core active window + vector pipes (mte2_aiv, vec, mte3)
    The vector unit splits into AIV0/AIV1 doing equal halves IN PARALLEL, so the AIV stage
    wall time is ONE core's window (we read AIV0), not the AIV0+AIV1 sum.
    """
    p = (csv_dir or CSV_DIR) / f"{fid}_pipeline_summary.csv"
    total = None
    aic = None
    aiv = None
    with p.open() as f:
        for r in csv.DictReader(f):
            total = int(r["total_cycles"])
            if r["unit"] == "AIC":
                aic = dict(active=int(r["active_cycles"]), busy=int(r["busy_cycles"]),
                           start=int(r["active_start_cycle"]), end=int(r["active_end_cycle"]),
                           mte2=int(r["mte2_aic_cycles"]), mte1=int(r["mte1_cycles"]),
                           cube=int(r["cube_cycles"]), fixp=int(r["fixp_cycles"]))
            elif r["unit"] == "AIV0":
                aiv = dict(active=int(r["active_cycles"]), busy=int(r["busy_cycles"]),
                           start=int(r["active_start_cycle"]), end=int(r["active_end_cycle"]),
                           mte2=int(r["mte2_aiv_cycles"]), vec=int(r["vec_cycles"]),
                           mte3=int(r["mte3_cycles"]))
    if total is None or aic is None or aiv is None:
        raise RuntimeError(f"missing AIC/AIV0 rows in {p}")
    return dict(total=total, aic=aic, aiv=aiv)


def overlap_factor(cube_wall, vec_wall, total):
    """How much of the smaller stage is hidden behind the larger, from per-UNIT continuous wall
    times: 1.0 = full overlap (total == max), 0.0 = serial (total == cube + vec == sum). Clamped
    to [0, 1]:  result = (cube_wall + vec_wall - total) / min(cube_wall, vec_wall).

    IMPORTANT -- which per-unit number to pass:
      * Pass each unit's ACTIVE SPAN measured in the OVERLAP run (active_end - active_start), where
        the unit runs continuously -> that span IS its true wall time.
      * Do NOT pass `busy_cycles`: busy SUMS a unit's pipes (mte2/mte1/cube/fixp), which pipeline
        internally, so busy OVER-counts the unit's wall (total < busy is normal).
      * Do NOT pass the span from the SERIAL run: there the two units interleave with idle gaps, so
        each span covers nearly the whole timeline and the metric collapses.
    The serial total exceeds cube_wall+vec_wall (it also loses intra-unit cross-tile pipelining),
    so the serial result clamps to 0.
    """
    lo = min(cube_wall, vec_wall)
    if lo == 0:
        return 0.0
    f = (cube_wall + vec_wall - total) / lo
    return max(0.0, min(1.0, f))


# --- testcase emitter (kernels self-contained in main.cpp; cube half includes RunGemmE2E) ---
CMAKE_TEMPLATE = (
    "pto_costmodel_sim_st({name})\n"
    "target_include_directories({name} PRIVATE {gemm_inc})\n"
    "target_compile_options({name} PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)\n"
)

MAIN_HEADER = """// AUTO-GENERATED by mixed_tile_study (run.py). Do not edit by hand.
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>
#include "gemm_performance_kernel.cpp"

using namespace pto;
"""


def write_testcase(name, fn_defs, fids, test_suite, launch_cfgs=None):
    """Write testcase/<name>/{main.cpp, CMakeLists.txt}. Mixed kernels are inline in fn_defs;
    the cube half reuses RunGemmE2E from gemm_performance_kernel.cpp (included via MAIN_HEADER).
    launch_cfgs: {fid: "(block_dim, nullptr, nullptr)"}.
    """
    launch_cfgs = launch_cfgs or {}
    tc = TESTCASE_DIR / name
    tc.mkdir(parents=True, exist_ok=True)
    body = [MAIN_HEADER]
    body += fn_defs
    body.append(f"\nTEST({test_suite}, All) {{")
    body += [f"    LAUNCH_KERNEL({fid}, , {launch_cfgs.get(fid, '(1, nullptr, nullptr)')});"
             for fid in fids]
    body.append("}")
    (tc / "main.cpp").write_text("\n".join(body) + "\n")
    (tc / "CMakeLists.txt").write_text(CMAKE_TEMPLATE.format(name=name, gemm_inc=GEMM_KERNEL_INCLUDE))
