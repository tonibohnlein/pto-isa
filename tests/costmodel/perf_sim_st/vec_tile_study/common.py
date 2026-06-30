# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Shared constants, the analytic vector-core cost model, and a perf-sim CSV reader for
# the VECTOR-tile cost-model study. See README.md for the full write-up.
#
# This is the *sibling* of gm_l1_tile_study (GM<->L1 cube reload) and l0_tile_study
# (L1->L0 cube extract). Where those scope the CUBE cores, THIS study scopes the VECTOR
# cores: the per-op compute formula (vadd/vmul/vexp/vreducev2 = slope*repeat + startup),
# the GM<->UB roofline, and the parallelism/streaming mechanisms. It grounds the mlsys26
# vector cost model against pto-isa's device-calibrated per-instruction stubs
# (include/pto/costmodel/a2a3/cce_costmodel/cce_costmodel_vector_compute.hpp).

import csv
import pathlib

# --- paths (self-contained; results land under results/) ---
STUDY_DIR = pathlib.Path(__file__).resolve().parent
PERF_SIM_ROOT = STUDY_DIR.parent                      # tests/costmodel/perf_sim_st
TESTCASE_DIR = PERF_SIM_ROOT / "testcase"
RESULTS_DIR = STUDY_DIR / "results"
CSV_DIR = RESULTS_DIR / "perf_sim_output"             # where LAUNCH_KERNEL writes summaries

# --- a2a3 constants (pto-isa include/pto/costmodel/arch_config.hpp) ---
FREQ_HZ = 1.85e9
BW_GM_UB = 100.9      # GB/s  (GM -> UB, the vector load port / MTE2_AIV)
BW_UB_GM = 188.46     # GB/s  (UB -> GM, the vector store port / MTE3)
VEC_REG_BYTES = 256   # a2a3 vector register: 64 fp32 / 128 fp16 elements per SIMD repeat

# --- pto-isa device-calibrated per-instruction VECTOR compute model -----------------
# cce_costmodel_core.hpp:133 EstimateLinearCycles(repeat, head, slope, tail):
#   isolated op  = slope*repeat + head + tail        (VECTOR pipe, queue empty at start)
#   in a stream  = slope*repeat                       (head+tail paid ONCE per stream;
#                  back-to-back vec ops overlap their startup latency away)
#   binary ALU   + 16 (kCountModeFloorCycles) when the tile is NOT repeat-aligned
# Per-op coefficients (910B3 标定, R^2~1.0) from cce_costmodel_vector_compute.hpp, as
# (slope, head+tail). The OP selector int matches the kernel's `if constexpr (OP==...)`.
VEC_OPS = {
    0: dict(name="add", instr="vadd",      slope=2,  ht=24, binary=True),   # 10,2,14
    1: dict(name="mul", instr="vmul",      slope=2,  ht=25, binary=True),   # 10,2,15
    2: dict(name="div", instr="vdiv",      slope=4,  ht=30, binary=True),   # 11,4,19
    3: dict(name="exp", instr="vexp",      slope=2,  ht=31, binary=False),  # 13,2,18 (unary)
}
COUNT_MODE_FLOOR = 16   # kCountModeFloorCycles (binary ALU, unaligned cols)

# mlsys26's current vector model (set_910b): ONE slope per class, ONE head+tail, per-op.
MLSYS_SLOPE_PW = 2.0
MLSYS_SLOPE_REDUCE = 14.0
MLSYS_HEAD = 14.0
MLSYS_TAIL = 18.0


def ceil_div(a, b):
    return (a + b - 1) // b


def repeat_for(rows, cols, bytes_t=4):
    """SIMD repeat count for a VL-aligned Vec tile: rows*cols / (256/bytes_t).

    One fused vector call over the whole tile (rows merged), valid when cols is a multiple
    of elements-per-repeat (64 fp32 / 128 fp16) and the total is <= REPEAT_MAX (255).
    """
    epr = VEC_REG_BYTES // bytes_t
    return rows * cols // epr


def perfsim_chain_cycles(op_sel, repeat, nops):
    """pto-isa perf-sim VECTOR cycles for a homogeneous NOPS-long stream of `op_sel`.

    = (head+tail paid ONCE) + nops * slope * repeat. The mlsys26 model instead charges
    (head + slope*repeat + tail) PER op -- the per-stream-vs-per-op gap this study measures.
    """
    o = VEC_OPS[op_sel]
    return o["ht"] + nops * o["slope"] * repeat


def mlsys_chain_cycles(repeat, nops, slope=MLSYS_SLOPE_PW):
    """mlsys26's current prediction: nops * (head + slope*repeat + tail) -- per-op startup."""
    return nops * (MLSYS_HEAD + slope * repeat + MLSYS_TAIL)


def mlsys_reduce_cycles(rows, cols, bytes_t=4):
    """mlsys26 charges a reduction as ONE op: head + slope_reduce*repeat + tail, with
    repeat = rows*cols / (256/bytes_t). So it grows with BOTH rows and cols.
    """
    return MLSYS_HEAD + MLSYS_SLOPE_REDUCE * repeat_for(rows, cols, bytes_t) + MLSYS_TAIL


# Per-op pieces of a barrier-isolated reduction pass (count-mode -> repeat=0, so slope*repeat
# vanishes): each pass = mask wraps + the op (head+tail [+count floor]) + pipe_barrier, and the
# barrier resets the VEC queue so EVERY pass re-pays startup. From cce_costmodel_vector_compute
# (vadd head+tail=24, vcadd head+tail=46) + EstimateCountModeFloor=16 + const mask/barrier cycles.
VADD_PASS = 24 + COUNT_MODE_FLOOR + 5    # count-mode vadd (40) + 5 const (mask*2/norm/mask/barrier)
VCADD_PASS = 46 + 5                      # final cross-lane vcadd block


def perfsim_trowsum_cycles(cols, bytes_t=4):
    """Predicted perf-sim TROWSUM cost: a binary tree of K-1 barrier-isolated count-mode vadd
    passes + a final vcadd, K = cols/(256/bytes_t). COUNT mode forces repeat=0, so the cost is
    ROWS-INDEPENDENT and ~linear in K -- structurally unlike mlsys26's slope_reduce*rows*cols.
    """
    epr = VEC_REG_BYTES // bytes_t
    k = cols // epr
    return VCADD_PASS if k < 2 else VADD_PASS * (k - 1) + VCADD_PASS


def perfsim_tcolsum_cycles(rows):
    """Predicted perf-sim TCOLSUM (binary) cost: the pairwise vadd tree across rows STREAMS
    within each level (one barrier per level), so only log2(R) startups are paid:
    ~16*(R-1) streamed count-mode vadds + 30*log2(R) per-level startup. Scales with the
    REDUCED dim ROWS -- unlike TROWSUM (reduce W) which is ROWS-independent. (R a power of 2.)
    """
    if rows < 2:
        return 0
    levels = rows.bit_length() - 1   # log2(R) for powers of 2
    return 16 * (rows - 1) + 30 * levels


def transfer_cycles(byts, bw_gibs):
    """Perf-sim memory-pipe busy cycles for `byts` at `bw_gibs` GiB/s (flat model).
    Mirrors EstimateBandwidthCycles: bytes / 2**30 / bw * freq_hz.
    """
    return byts / (1024.0 ** 3) / bw_gibs * FREQ_HZ


def read_aiv(fid, csv_dir=None):
    """One AIV sub-core's pipe busy cycles for kernel function `fid`.

    The vector unit splits into AIV0/AIV1 (equal per-core work); we read AIV0. Returns
    vec (vec_cycles, the VECTOR compute pipe), mte2 (mte2_aiv = GM->UB load), mte3
    (UB->GM store), and total.
    """
    p = (csv_dir or CSV_DIR) / f"{fid}_pipeline_summary.csv"
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["unit"] == "AIV0":
                return dict(total=int(r["total_cycles"]), vec=int(r["vec_cycles"]),
                            mte2=int(r["mte2_aiv_cycles"]), mte3=int(r["mte3_cycles"]))
    raise RuntimeError(f"no AIV0 row in {p}")


# --- testcase emitter (self-contained kernels inline in main.cpp) ---
CMAKE_TEMPLATE = (
    "pto_costmodel_sim_st({name})\n"
    "target_compile_options({name} PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)\n"
)

MAIN_HEADER = """// AUTO-GENERATED by vec_tile_study (run.py). Do not edit by hand.
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>

using namespace pto;
"""


def write_testcase(name, fn_defs, fids, test_suite, launch_cfgs=None):
    """Write testcase/<name>/{main.cpp, CMakeLists.txt}. Kernels are self-contained in
    fn_defs (no external kernel header). launch_cfgs: {fid: "(block_dim, nullptr, nullptr)"}.
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
    (tc / "CMakeLists.txt").write_text(CMAKE_TEMPLATE.format(name=name))
