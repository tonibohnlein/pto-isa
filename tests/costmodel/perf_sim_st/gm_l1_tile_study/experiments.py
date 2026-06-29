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


# ---------------------------------------------------------------------------- splitk
# Split-K (sink) as the per-core workload. A parallel split-K sink launches S workers,
# each computing a FULL M*N partial over a K/S contraction slice, then atomic-adding it
# to GM. RunGemmE2E with k=Kc=K/S IS one such worker (the atomic-add store has the same
# L0C->GM byte volume as a plain TSTORE, so the plain store is a faithful proxy).
# Sweeping Kc shows feed (MTE2) and compute (CUBE) shrink ~ Kc while the output store
# (FixPipe) is a CONSTANT floor -- exactly the trade-off the mlsys26 eval_S enumeration
# optimizes: splitting helps until the per-core wall hits that store floor. (The UPWARD
# re-inflation at large S -- aggregate S*store saturating HBM via par() -- is multi-core
# and not visible in the single-core AIC row; see DESIGN_SPACE.md.)
def gen_splitk():
    M = N = 512
    K = 1024
    bm = bn = 128
    bk = 64
    assert _fits(bm, bk, bn)
    defs, fids, index = [], [], []
    for S in (1, 2, 4, 8, 16):
        Kc = K // S
        if Kc < bk or Kc % bk:
            continue
        fid = f"sp{S}"
        defs.append(C.emit_e2e(fid, M, Kc, N, bm, bk, bn))
        fids.append(fid)
        index.append(dict(id=fid, M=M, N=N, K=K, S=S, Kc=Kc, bm=bm, bk=bk, bn=bn))
    C.write_testcase("gml1_splitk", "gemm_performance_kernel.cpp", defs, fids, "Gml1SplitK")
    return _save_index("splitk", index)


# ----------------------------------------------------------------------------- chain
# Chained matmul (C = A*B, E = C*D): validates the model's produced-operand EXCLUSION.
# cube_operand_reload() walks the {MM1,MM2} subgraph and charges GM reload only for
# BOUNDARY operands (A,B,D); the intermediate C is `produced` on-chip and never hits
# DDR. We can't keep C on-chip with the single-matmul RunGemmE2E, so we measure MM1 and
# MM2 SEPARATELY (two single-core runs) and decompose: the C round-trip that fusion
# eliminates is exactly MM1's C-store (fixp) + MM2's C-reload (the lhs half of its mte2).
# Sweeping the shared dim Ki=N1=K2 shows the round-trip (the fusion saving) scale with it
# while the boundary reloads (A,B,D) stay put. This validates the cost-model accounting;
# the fused lowering (C resident in L1) is a separate lowering concern, not scored here.
def gen_chain():
    M = 512        # rows of A, C, E
    K1 = 512       # contraction of MM1 (cols of A)
    N2 = 512       # cols of D, E
    bm = bn = 128
    bk = 64
    assert _fits(bm, bk, bn)
    defs, fids, index = [], [], []
    for Ki in (128, 256, 512):     # shared dim: N1 (cols of C) = K2 (contraction of MM2)
        if Ki % bn or Ki % bk:
            continue
        # MM1: A[M,K1] * B[K1,Ki] -> C[M,Ki]
        f1 = f"ch{Ki}_mm1"
        defs.append(C.emit_e2e(f1, M, K1, Ki, bm, bk, bn))
        # MM2: C[M,Ki] * D[Ki,N2] -> E[M,N2]
        f2 = f"ch{Ki}_mm2"
        defs.append(C.emit_e2e(f2, M, Ki, N2, bm, bk, bn))
        fids += [f1, f2]
        index.append(dict(Ki=Ki, M=M, K1=K1, N2=N2, bm=bm, bk=bk, bn=bn, mm1=f1, mm2=f2))
    C.write_testcase("gml1_chain", "gemm_performance_kernel.cpp", defs, fids, "Gml1Chain")
    return _save_index("chain", index)


# ----------------------------------------------------------------------------- fused
# Truly fused chain (chain_fused_kernel.cpp): C = A*B kept in L1, E = C*D from L1.
# Confirms DIRECTLY (not by decomposition) that the intermediate C never round-trips GM:
# fused MTE2 = reload(A,B,D), with NO C reload. Single M row-band, Ki one L1 tile. We also
# emit the matching unfused pair (two RunGemmE2E) so the C-reload saving is read off.
def gen_fused():
    cases = [  # (M, K1, Ki, N2, bm, bk, bnE)  -- Ki == one C tile (cL0/cAcc <= L0); M/bm bands
        (128, 256, 128, 256, 128, 64, 64),   # 1 band (M == bm)
        (128, 512, 128, 512, 128, 64, 64),
        (128, 512, 256, 512, 128, 64, 64),
        (256, 512, 128, 512, 128, 64, 64),   # 2 bands
        (512, 512, 128, 512, 128, 64, 64),   # 4 bands -- B,D reloaded per band (M/bm)
    ]
    defs, fids, index = [], [], []
    for (M, K1, Ki, N2, bm, bk, bnE) in cases:
        assert M % bm == 0 and bm * Ki * 4 <= L0C and bm * Ki * 2 <= L0A, (M, Ki)
        ff = f"fz_{M}_{K1}_{Ki}_{N2}"
        defs.append(f"void {ff}() {{ gm_l1_chain::RunGemmChainFused<float, half, half, "
                    f"{M}, {K1}, {Ki}, {N2}, {bm}, {bk}, {bnE}>(nullptr, nullptr, nullptr, nullptr); }}")
        u1, u2 = f"{ff}_mm1", f"{ff}_mm2"
        defs.append(C.emit_e2e(u1, M, K1, Ki, bm, bk, Ki))     # unfused MM1: A*B -> C[M,Ki]
        defs.append(C.emit_e2e(u2, M, Ki, N2, bm, Ki, bnE))    # unfused MM2: C*D -> E (bk=Ki)
        fids += [ff, u1, u2]
        index.append(dict(M=M, K1=K1, Ki=Ki, N2=N2, bm=bm, bk=bk, bnE=bnE, fused=ff, mm1=u1, mm2=u2))
    C.write_testcase("gml1_fused", "chain_fused_kernel.cpp", defs, fids, "Gml1Fused")
    return _save_index("fused", index)


# ------------------------------------------------------------------------- multicore
# Multi-core aggregate / par(): the perf-sim's Hill total_read_gibs cap is exactly the
# mlsys26 par(active, peak) = min(active, hbm/peak). gemm_performance partitioned along N
# (singleCoreN = N/B) runs on B cores; LAUNCH_KERNEL sets SetActiveCoreCount(B) and the
# capped fids set total_read_gibs = HBM, so per-core BwEff = min(peak, HBM/B). Uncapped:
# per-core MTE2 ~ 1/B (linear, par = active -- mlsys26's disabled 3240 cap). Capped:
# per-core MTE2 plateaus once B > HBM/peak (aggregate saturates at HBM) -- par() live.
def gen_multicore():
    M = N = 2048
    K = 512
    bm = bn = 128
    bk = 64
    hbm = C.HBM_AGGREGATE_GIBS
    defs, fids, index, cfgs = [], [], [], {}

    def emit(fid, B, cap):
        scn = N // B   # partition the output columns across B cores
        setup = ("auto _m = pto::mocker::evaluator::MakeFlatHillModel(); "
                 f"_m.total_read_gibs = {cap}; pto::mocker::evaluator::SetHillBandwidthModel(_m); ")
        call = (f"RunGemmE2E<float, half, half, float, {B}, {M}, {K}, {N}, {M}, {K}, {N}, "
                f"{M}, {K}, {scn}, {bm}, {bk}, {bn}, 1, 1, 1, 1>(nullptr, nullptr, nullptr);")
        defs.append(f"void {fid}() {{ {setup}{call} }}")
        fids.append(fid)
        cfgs[fid] = f"({B}, nullptr, nullptr)"

    for B in (1, 2, 4, 8, 16):
        if N % B or (N // B) % bn:
            continue
        emit(f"mc_un_{B}", B, 0.0)      # uncapped: total_read = 0 -> no contention
        emit(f"mc_cap_{B}", B, hbm)     # capped:   total_read = HBM -> par() saturation
        index.append(dict(B=B, M=M, N=N, K=K, bm=bm, bk=bk, bn=bn, hbm=hbm,
                          un=f"mc_un_{B}", cap=f"mc_cap_{B}"))
    C.write_testcase("gml1_multicore", "gemm_performance_kernel.cpp", defs, fids,
                     "Gml1Multicore", launch_cfgs=cfgs)
    return _save_index("multicore", index)


ALL = {
    "gml1_reload": gen_reload,
    "gml1_roofline": gen_roofline,
    "gml1_stepk": gen_stepk,
    "gml1_splitk": gen_splitk,
    "gml1_chain": gen_chain,
    "gml1_fused": gen_fused,
    "gml1_multicore": gen_multicore,
}
