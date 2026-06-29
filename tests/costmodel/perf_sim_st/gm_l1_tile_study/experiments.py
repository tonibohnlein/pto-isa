# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Experiment definitions for the GM->L1-tile cost-model study. Each generator writes a
# perf-sim testcase (testcase/<name>/{main.cpp,CMakeLists.txt}) and returns an index of
# (kernel-fn -> config) used by analyze.py. All run on a SINGLE core (blockDim=1) so the
# AIC pipeline summary measures exactly the per-core reload our mlsys26 model scores.
# See README.md. NOTE: testcase names here must also be listed in testcase/CMakeLists.txt.

import json

import common as C

L0A, L0B, L0C, PING, L1 = C.L0A, C.L0B, C.L0C, C.L0_PING, C.L1


def _save_index(name, index):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / f"{name}_index.json").write_text(json.dumps(index, indent=2))
    return index


def _fits(bm, bk, bn, stepKa=1, stepKb=1):
    """Legality: L0A/L0B ping-pong slots, single L0C, and double-buffered L1 staging."""
    l0a = bm * bk * 2 <= PING
    l0b = bk * bn * 2 <= PING
    l0c = bm * bn * 4 <= L0C
    l1 = 2 * (bm * bk * stepKa + bk * bn * stepKb) * 2 <= L1   # BUFFER_NUM=2
    return l0a and l0b and l0c and l1


# --------------------------------------------------------------------------- reload
# The core validation: does the perf-sim MTE2 (GM->L1) cycle count match our model's
# reload = MNK/bn*ba + MNK/bm*bb across a (bm, bn) tile sweep? Confirms BOTH the
# reload byte formula AND the effective GM->L1 bandwidth (flat 135 GiB/s hypothesis).
def gen_reload():
    M = N = K = 512
    bk = 64
    defs, fids, index = [], [], []
    n = 0
    for bm in C.divisors(M):
        for bn in C.divisors(N):
            if not _fits(bm, bk, bn):
                continue
            fid = f"rl{n}"
            defs.append(C.emit_e2e(fid, M, K, N, bm, bk, bn))
            fids.append(fid)
            index.append(dict(id=fid, M=M, N=N, K=K, bm=bm, bk=bk, bn=bn))
            n += 1
    C.write_testcase("gml1_reload", "gemm_performance_kernel.cpp", defs, fids, "Gml1Reload")
    return _save_index("reload", index)


# -------------------------------------------------------------------------- roofline
# Regime sweep: across memory-bound (skinny, small K, big MN) -> compute-bound (square,
# deep K), does total ~= max(mte2, mte1, cube, fixp)? Validates our max-roofline AND the
# FixPipe overlap (feed GM->L1 vs store L0C->GM are SEPARATE pipes: total ~= max, not sum).
def gen_roofline():
    problems = [  # (label, M, N, K, bm, bk, bn)
        ("skinny_membound", 1024, 1024, 128, 128, 64, 128),
        ("smalltile_reload", 512, 512, 256, 64, 64, 64),
        ("balanced", 512, 512, 512, 128, 64, 128),
        ("deepk_compute", 256, 256, 2048, 128, 64, 128),
        ("square_compute", 512, 512, 1024, 128, 64, 128),
    ]
    defs, fids, index = [], [], []
    for (label, M, N, K, bm, bk, bn) in problems:
        assert M % bm == 0 and N % bn == 0 and K % bk == 0, label
        assert _fits(bm, bk, bn), f"{label} does not fit L0/L1"
        fid = f"rf_{label}"
        defs.append(C.emit_e2e(fid, M, K, N, bm, bk, bn))
        fids.append(fid)
        index.append(dict(id=fid, label=label, M=M, N=N, K=K, bm=bm, bk=bk, bn=bn))
    C.write_testcase("gml1_roofline", "gemm_performance_kernel.cpp", defs, fids, "Gml1Roofline")
    return _save_index("roofline", index)


# ----------------------------------------------------------------------------- stepk
# K-staging invariance: stepKa/stepKb batch the TLOADs into larger L1 panels but do NOT
# change the total GM->L1 byte volume. Validates that our model correctly OMITS a stepK
# term (mte2 cycles invariant), while bigger staging changes only overlap (total).
def gen_stepk():
    M = N = K = 512
    bm = bn = 128
    bk = 64
    defs, fids, index = [], [], []
    for step in (1, 2, 4):
        if not _fits(bm, bk, bn, step, step):
            continue
        if K % (bk * step):
            continue
        fid = f"sk{step}"
        defs.append(C.emit_e2e(fid, M, K, N, bm, bk, bn, stepKa=step, stepKb=step))
        fids.append(fid)
        index.append(dict(id=fid, M=M, N=N, K=K, bm=bm, bk=bk, bn=bn, step=step))
    C.write_testcase("gml1_stepk", "gemm_performance_kernel.cpp", defs, fids, "Gml1StepK")
    return _save_index("stepk", index)


ALL = {
    "gml1_reload": gen_reload,
    "gml1_roofline": gen_roofline,
    "gml1_stepk": gen_stepk,
}
