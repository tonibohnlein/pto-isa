# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Shared constants, the analytic a2a3 GM->L1 cost model, and a perf-sim CSV reader
# for the GM->L1-tile cost-model study. See README.md for the full write-up.
#
# This is the *sibling* study to l0_tile_study/. Where that study scopes the L1->L0
# boundary (compute-bound, operands L1-resident, validates the CUBE/MTE1 pipes),
# THIS study scopes the GM<->L1 boundary (memory-bound, operands streamed from HBM,
# validates the MTE2 / GM->L1 reload pipe + the FixPipe drain + the max-roofline).
# It reuses the same gemm_performance reference kernel (RunGemmE2E), whose TLOADs
# (GM->L1, MatTile) are exactly the operand reload our mlsys26 cube model predicts.

import csv
import pathlib

# --- paths (self-contained; results land under results/) ---
STUDY_DIR = pathlib.Path(__file__).resolve().parent
PERF_SIM_ROOT = STUDY_DIR.parent                      # tests/costmodel/perf_sim_st
TESTCASE_DIR = PERF_SIM_ROOT / "testcase"
KERNEL_INCLUDE = "../../../kernels/manual/a2a3/gemm_performance"  # for CMake include dir
RESULTS_DIR = STUDY_DIR / "results"
CSV_DIR = RESULTS_DIR / "perf_sim_output"             # where LAUNCH_KERNEL writes summaries
CSV_DIR_FITTED = RESULTS_DIR / "fitted" / "perf_sim_output"  # PTO_BW_MODE=fitted run (run.py --fitted)

# --- a2a3 cost-model constants (pto-isa include/pto/costmodel/arch_config.hpp) ---
# BandwidthTable field order is the source of truth for these (see arch_config.hpp:46).
FREQ_HZ = 1.85e9
BW_GM_L1 = 135.0      # GB/s  (GM -> L1, the MTE2 reload port; flat/legacy a2a3 value)
BW_GM_UB = 100.9      # GB/s  (GM -> UB, vector reload)
BW_L1_GM = 32.0       # GB/s  (L1 -> GM)
BW_L0C_GM = 70.0      # GB/s  (FIXPIPE drain L0C -> GM, the matmul output store)
BW_L0C_L1 = 128.0     # GB/s  (FIXPIPE drain L0C -> L1)
BW_L1_L0A = 441.0     # GB/s  (L1 -> L0A, the cube's A/"left" port)
BW_L1_L0B = 220.5     # GB/s  (L1 -> L0B, the B/"right" port; exactly half of A)

# Fitted (on-device) GM->L1 Hill params (arch_config.hpp MakeFittedHillModel, PTO_BW_MODE=fitted):
#   HillBw(bytes) = 28.61 * bytes / (1107 + bytes)   -> saturates at 28.61 GiB/s, ~4.7x below flat.
# Our mlsys26 cube model hardcodes bw_gm_l1 = 135 (flat); this study quantifies that gap.
BW_GM_L1_FITTED_PEAK = 28.61
BW_GM_L1_FITTED_K = 1107.0

# Aggregate HBM read bandwidth (GiB/s) for the multi-core par() experiment. mlsys26's
# hbm_aggregate_gibps = 24*135 = 3240 effectively disables the cap (par = active); the
# realistic A3 figure is ~900, which saturates at 900/135 ~= 6.7 cores. The perf-sim's
# Hill total_read_gibs models exactly this aggregate cap (BwEff divides peak by ncores).
HBM_AGGREGATE_GIBS = 900.0

# --- buffer capacities (bytes) ---
L0A = L0B = 64 * 1024
L0C = 128 * 1024
L0_PING = 32 * 1024   # per ping-pong slot when an operand buffer is double-buffered
L1 = 512 * 1024       # a2a3 L1 capacity

BYTES = {"bf16": 2, "fp32": 4}


def ceil_div(a, b):
    return (a + b - 1) // b


def divisors(d, align=16, lo=16):
    return [x for x in range(lo, d + 1, align) if d % x == 0]


def reload_bytes(M, N, K, bm, bn, bytes_a=2, bytes_b=2):
    """GM->L1 operand reload volume for one core, output-stationary (RunGemmE2E).

    Our mlsys26 cube model's cube_operand_reload():
        reload = M*N*K/bn * bytes_a   (A panel reloaded once per N-block: N/bn times)
               + M*N*K/bm * bytes_b   (B panel reloaded once per M-block: M/bm times)
    This is exactly the byte volume RunGemmE2E's TLOADs issue (the K-staging knobs
    stepKa/stepKb only batch the TLOADs, they do NOT change the total bytes).
    """
    return M * N * K / bn * bytes_a + M * N * K / bm * bytes_b


def store_bytes(M, N, bytes_c=2):
    """L0C->GM matmul output store (the FixPipe drain), shape-only.

    NOTE: the perf-sim's copy_matrix_cc_to_gm charges L0C_TO_GM (70 GiB/s) on a
    *2-byte* (bf16) drain even though the L0C accumulator is fp32 -- the FixPipe
    casts fp32->bf16 on the way out. So the drain width is the OUTPUT dtype (2 B),
    not the accumulator (4 B). Measured: fixp = M*N*2/70 to <0.1% (512^2 and 1024^2).
    mlsys26's out_store uses dtype_bytes(output tensor) -- correct iff that dtype is bf16.
    """
    return M * N * bytes_c


def transfer_cycles(byts, bw_gibs):
    """Perf-sim memory-pipe busy cycles for `byts` at `bw_gibs` GiB/s (flat model).

    Mirrors EstimateBandwidthCycles: bytes / 2**30 / bw * freq_hz. The flat a2a3
    table value (e.g. BW_GM_L1=135) is interpreted as GiB/s here.
    """
    return byts / (1024.0 ** 3) / bw_gibs * FREQ_HZ


def hill_bw_gibs(byts, peak=BW_GM_L1_FITTED_PEAK, k=BW_GM_L1_FITTED_K):
    """Fitted Hill GM->L1 bandwidth (GiB/s) for ONE transfer of `byts` bytes:
    HillBw(B) = peak * B / (k + B). Saturates at `peak`; small B is penalised by `k`.
    """
    return peak * byts / (k + byts) if (k + byts) > 0 else peak


def fitted_tload_cycles(byts):
    """Perf-sim MTE2 cycles for ONE GM->L1 TLOAD of `byts` under the fitted Hill model.
    = (k + byts) * freq / (2**30 * peak) -- a per-transfer FIXED cost (k) + bandwidth term.
    """
    if byts <= 0:
        return 0.0
    return byts / (1024.0 ** 3) / hill_bw_gibs(byts) * FREQ_HZ


def e2e_tloads(M, N, K, bm, bk, bn, stepKa=1, stepKb=1, bytes_a=2, bytes_b=2):
    """(count, bytes-per-transfer) of the A and B GM->L1 TLOADs RunGemmE2E issues.

    A panel [bm, bk*stepKa] is loaded once per (i,j) every stepKa K-iters; B panel
    [bk*stepKb, bn] likewise. Total bytes reduce to reload_bytes(); under the FLAT model
    only the total matters, but under the FITTED Hill model the per-transfer SIZE matters
    (the `k` floor penalises small TLOADs), so the granularity (tile, stepK) is exposed.
    """
    tiles = (M // bm) * (N // bn)
    a = (tiles * (K // (bk * stepKa)), bm * bk * stepKa * bytes_a)
    b = (tiles * (K // (bk * stepKb)), bk * stepKb * bn * bytes_b)
    return [a, b]


def fitted_reload_cycles(M, N, K, bm, bk, bn, stepKa=1, stepKb=1):
    """Predicted MTE2 cycles for RunGemmE2E under PTO_BW_MODE=fitted (sum over TLOADs)."""
    return sum(cnt * fitted_tload_cycles(b) for cnt, b in e2e_tloads(M, N, K, bm, bk, bn, stepKa, stepKb))


def mad_cycles(m, k, n, bytes_a=2):
    """Cube MAD cost (pto-isa formula_backend_compute.hpp): one TMATMUL call."""
    kt = 32 // bytes_a            # 16 (bf16) / 8 (fp32)
    cpr = 2 if bytes_a == 4 else 1
    return 6 + cpr * ceil_div(m, 16) * ceil_div(k, kt) * ceil_div(n, 16)


def cube_cycles(M, N, K, bm, bk, bn, bytes_a=2):
    """Total cube cycles over the (M/bm)(N/bn)(K/bk) tile grid."""
    return (M // bm) * (N // bn) * (K // bk) * mad_cycles(bm, bk, bn, bytes_a)


def mte1_cycles(M, N, K, bm, bk, bn, bytes_a=2, bytes_b=2):
    """L1->L0 extract cost (the cube's MTE1 pipe): A streams through L0A (BW 441), B
    through L0B (BW 220.5). A re-extracts with N-tiling (1/bn), B with M-tiling (1/bm) --
    so the port ASYMMETRY makes tall tiles (big bm) cheaper: the slow L0B port carries B,
    and big bm cuts B re-extracts. This is the L0-study term the GM->L1 roofline omits.
    """
    a = transfer_cycles(M * N * K * bytes_a / bn, BW_L1_L0A)
    b = transfer_cycles(M * N * K * bytes_b / bm, BW_L1_L0B)
    return a + b


def read_aic(fid, csv_dir=None):
    """Return the AIC-row pipe busy cycles for kernel function `fid`.

    csv_dir defaults to the flat-model CSV_DIR; pass CSV_DIR_FITTED for the fitted run.
    """
    p = (csv_dir or CSV_DIR) / f"{fid}_pipeline_summary.csv"
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["unit"] == "AIC":
                return dict(
                    total=int(r["total_cycles"]),
                    mte2=int(r["mte2_aic_cycles"]),
                    mte1=int(r["mte1_cycles"]),
                    cube=int(r["cube_cycles"]),
                    fixp=int(r["fixp_cycles"]),
                )
    raise RuntimeError(f"no AIC row in {p}")


def read_aiv(fid, csv_dir=None):
    """Return one AIV sub-core's pipe busy cycles for kernel function `fid`.

    The vector unit splits into two sub-cores (AIV0/AIV1) under each AIC; they carry equal
    per-core work, so we read AIV0. mte2 here is mte2_aiv_cycles (GM->UB, the vector read
    pipe), the sibling of the AIC's mte2_aic_cycles (GM->L1). Both draw from total_read_gibs.
    """
    p = (csv_dir or CSV_DIR) / f"{fid}_pipeline_summary.csv"
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["unit"] == "AIV0":
                return dict(total=int(r["total_cycles"]), mte2=int(r["mte2_aiv_cycles"]),
                            vec=int(r["vec_cycles"]), mte3=int(r["mte3_cycles"]))
    raise RuntimeError(f"no AIV0 row in {p}")


# RunGemmE2E<float,half,half,float, blockDim, m,k,n, valid..., singleCore..., base..., steps>
# bf16 operands, fp32 accumulate -- the autotiler's default GEMM dtypes. Single core
# (blockDim=1, singleCore = whole problem) isolates the per-core reload our model scores.
def emit_e2e(fid, M, K, N, bm, bk, bn, stepKa=1, stepKb=1):
    """A single-core GM->L1 GEMM call (the gemm_performance reference, RunGemmE2E)."""
    return (
        f"void {fid}() {{ RunGemmE2E<float, half, half, float, 1, "
        f"{M}, {K}, {N}, {M}, {K}, {N}, {M}, {K}, {N}, "
        f"{bm}, {bk}, {bn}, 1, {stepKa}, {stepKb}, 1>"
        f"(nullptr, nullptr, nullptr); }}"
    )


CMAKE_TEMPLATE = (
    "pto_costmodel_sim_st({name})\n"
    "target_include_directories({name} PRIVATE\n"
    "    ${{PROJECT_SOURCE_DIR}}/" + KERNEL_INCLUDE + "\n"
    ")\n"
    "target_compile_options({name} PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)\n"
)

MAIN_HEADER = """// AUTO-GENERATED by gm_l1_tile_study (run.py). Do not edit by hand.
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>

#include "{kernel_include}"

using namespace pto;
"""


def write_testcase(name, kernel_include, fn_defs, fids, test_suite, launch_cfgs=None):
    """Write testcase/<name>/{main.cpp, CMakeLists.txt}.

    launch_cfgs: optional {fid: "(block_dim, nullptr, nullptr)"} to run a kernel on
    multiple cores (default single core). LAUNCH_KERNEL reads block_dim and calls
    SetActiveCoreCount(block_dim), which drives the Hill model's per-core BW divide.
    """
    launch_cfgs = launch_cfgs or {}
    tc = TESTCASE_DIR / name
    tc.mkdir(parents=True, exist_ok=True)
    body = [MAIN_HEADER.format(kernel_include=kernel_include)]
    body += fn_defs
    body.append(f"\nTEST({test_suite}, All) {{")
    body += [f"    LAUNCH_KERNEL({fid}, , {launch_cfgs.get(fid, '(1, nullptr, nullptr)')});"
             for fid in fids]
    body.append("}")
    (tc / "main.cpp").write_text("\n".join(body) + "\n")
    (tc / "CMakeLists.txt").write_text(CMAKE_TEMPLATE.format(name=name))
