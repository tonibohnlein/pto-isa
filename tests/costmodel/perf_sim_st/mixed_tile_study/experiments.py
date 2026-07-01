# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Experiment definitions for the MIXED cube+vector tile cost-model study. Each generator writes a
# perf-sim testcase (testcase/<name>/{main.cpp,CMakeLists.txt}) and returns an index used by
# analyze.py. The CUBE half reuses RunGemmE2E from gemm_performance_kernel.cpp (included by
# common.MAIN_HEADER); the VECTOR epilogue + the two pipeline schedules are inline below.
# NOTE: testcase names here must also be listed in testcase/CMakeLists.txt (ALL_TESTCASES).
#
# Two schedules over the SAME tiled matmul->pointwise work isolate the overlap question:
#   mixed_overlap  -- SKEWED ping-pong (the SkewCrossCorePipeline producer-skew): the cube runs
#                     one tile ahead on the OTHER of two GM buffers, so cube(k+1) overlaps
#                     vector(k).  total ~= max(cube, vec) + one tile of fill/drain.
#   mixed_serial   -- single handoff buffer + B-operand chaining: cube(k>=1) reads its B operand
#                     FROM the handoff buffer the prior vector wrote, so the perf-sim's RAW edge
#                     forces cube(k) to wait for vector(k-1).  total ~= cube + vec (the sum).
#
# Why the dep tricks work (see tile_dep_tracker.hpp): the perf-sim tracks ONLY read-after-write
# cross-pipe deps (an op's INPUT vs the latest writer of that address). WAR/WAW are NOT tracked.
# So overlap needs distinct producer/consumer buffers (ping-pong); serial needs a real RAW edge
# from the vector's output back into the next cube's INPUT (the B-operand chain).

import json

import common as C


def _save_index(name, index):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / f"{name}_index.json").write_text(json.dumps(index, indent=2))
    return index


# ── inline mixed kernels (cube tile via RunGemmE2E + vector epilogue + the two schedules) ──
_MIXED_KERNEL = r"""
// One [BM,N] output tile = A_block @ B  (fp16 in, fp32 acc), via the proven gemm kernel.
// baseK = BK (= min(K,128)) keeps the L0A/L0B ping-pong <= 32 KiB; singleCoreK = K, so the cube
// runs a kLoop = K/BK accumulate loop for K > 128. BK defaults to min(K,128) at the call sites.
template <int BM, int K, int N, int BK>
AICORE inline void CubeTile(__gm__ float *out, __gm__ half *a, __gm__ half *b) {
    RunGemmE2E<float, half, half, float, /*blockDim=*/1,
               BM, K, N, BM, K, N, BM, K, N, BM, BK, N, 1, 1, 1, 1>(out, a, b);
}

// In-place pointwise epilogue over the [BM,N] handoff tile: load GM->UB, op, store UB->GM. The
// TLOAD reads `cbuf` -> a cross-pipe RAW edge onto the cube FIX store that produced it; the
// TSTORE re-registers `cbuf` as freshly written (read by the next cube in the serial chain).
template <int OP, int BM, int N>
AICORE inline void VectorTile(__gm__ float *cbuf, std::size_t ub_off) {
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT        = pto::Tile<pto::TileType::Vec, float, BM, N, pto::BLayout::RowMajor, BM, N>;
    VT v;
    TASSIGN(v, ub_off);
    ShapeDyn  shape(1, 1, 1, BM, N);
    StrideDyn stride(BM * N, BM * N, BM * N, N, 1);
    Global g(cbuf, shape, stride);
    TLOAD(v, g);                                  // GM->UB (MTE2_AIV): waits the cube store on cbuf
    if constexpr (OP == 0) { TADD(v, v, v); }     // VEC: C = C + C
    else if constexpr (OP == 3) { TEXP(v, v); }   // VEC: C = exp(C)
    TSTORE(g, v);                                 // UB->GM (MTE3): re-registers cbuf's writer
}

// SKEWED ping-pong: the cube produces tile 0 up front, then each loop step produces tile k+1 into
// the OTHER of two GM buffers while the vector consumes tile k from THIS buffer. cube(k+1) and
// vector(k) touch different buffers -> no dep -> the two units overlap. The per-tile RAW (vector
// reads the cube's buffer) is hidden one tile deep. NTILES=1 degenerates to a single serial tile.
template <int OP, int BM, int K, int N, int NTILES, int BK = (K < 128 ? K : 128)>
AICORE inline void MixedOverlap(__gm__ half *A, __gm__ half *B, __gm__ float *buf0, __gm__ float *buf1) {
    __gm__ float *buf[2] = {buf0, buf1};
    CubeTile<BM, K, N, BK>(buf[0], A, B);                                  // prologue: produce tile 0
    for (int k = 0; k < NTILES; ++k) {
        if (k + 1 < NTILES) {
            CubeTile<BM, K, N, BK>(buf[(k + 1) & 1], A + (k + 1) * BM * K, B);  // produce tile k+1 (other buffer)
        }
        VectorTile<OP, BM, N>(buf[k & 1], 0x100000);                       // consume tile k (this buffer)
    }
}

// SERIAL: one handoff buffer H, reused every tile. cube(k>=1) takes its B operand FROM H (the
// buffer the prior vector wrote) -> a tracked RAW edge -> cube(k) cannot start until vector(k-1)
// finishes. vector(k) then waits cube(k) (RAW on H). The chain cube(0)->vec(0)->cube(1)->...
// is fully serialized: total ~= NTILES*(cube + vec).
template <int OP, int BM, int K, int N, int NTILES, int BK = (K < 128 ? K : 128)>
AICORE inline void MixedSerial(__gm__ half *A, __gm__ half *B, __gm__ float *H) {
    for (int k = 0; k < NTILES; ++k) {
        __gm__ half *bk = (k == 0) ? B : reinterpret_cast<__gm__ half *>(H);  // chain B<-H for k>=1
        CubeTile<BM, K, N, BK>(H, A + k * BM * K, bk);   // k>=1: the B-load on H waits vector(k-1)
        VectorTile<OP, BM, N>(H, 0x100000);              // reads H -> waits cube(k)
    }
}
"""

# 3-stage vec1 -> cube -> vec2 kernel (mixed_vcv). Extends the skew with a vec1 PRE-stage on the
# cube's left operand, so the AIV runs BOTH vector phases and the fill deepens to a 2-stage prologue.
_VCV_KERNEL = r"""
// vec1 PRE-op: A' = A + A over [BM,K] (half), read `src` -> write `dst`. An AIV (vector) op that
// feeds the cube's left operand; the cube then reads A' (RAW on this store) one stage downstream.
template <int OP, int BM, int K>
AICORE inline void VecPreOp(__gm__ half *src, __gm__ half *dst, std::size_t ub_off) {
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT        = pto::Tile<pto::TileType::Vec, half, BM, K, pto::BLayout::RowMajor, BM, K>;
    VT v;
    TASSIGN(v, ub_off);
    ShapeDyn  shape(1, 1, 1, BM, K);
    StrideDyn stride(BM * K, BM * K, BM * K, K, 1);
    Global gs(src, shape, stride);
    Global gd(dst, shape, stride);
    TLOAD(v, gs);                                 // read A  (GM->UB, MTE2_AIV)
    if constexpr (OP == 0) { TADD(v, v, v); }     // A' = A + A  (VEC)
    else if constexpr (OP == 3) { TEXP(v, v); }
    TSTORE(gd, v);                                // write A' (UB->GM, MTE3): cube reads this next
}

// SKEWED 3-stage: vec1(k) -> cube(k) -> vec2(k). vec1 and vec2 BOTH run on the AIV; the cube on the
// AIC. Both handoffs (A'-buf, C-buf) are ping-ponged, and the pipeline is skewed two deep so
// vec1[k+2] || cube[k+1] || vec2[k] overlap. vec2(k) can first start only after vec1(0)->cube(0)
// have BOTH landed, so the prologue fill deepens to vec1_tile + cube_tile (vs the 2-stage cube_tile).
template <int OP, int BM, int K, int N, int NTILES, int BK = (K < 128 ? K : 128)>
AICORE inline void MixedVCV(__gm__ half *A, __gm__ half *B, __gm__ half *ap0, __gm__ half *ap1,
                            __gm__ float *c0, __gm__ float *c1) {
    __gm__ half  *ap[2] = {ap0, ap1};
    __gm__ float *cb[2] = {c0, c1};
    VecPreOp<OP, BM, K>(A, ap[0], 0x200000);                              // vec1(0) -> ap[0]
    CubeTile<BM, K, N, BK>(cb[0], ap[0], B);                             // cube(0): ap[0] -> cb[0]
    if (NTILES > 1) VecPreOp<OP, BM, K>(A + BM * K, ap[1], 0x200000);     // vec1(1) -> ap[1]
    for (int k = 0; k < NTILES; ++k) {
        VectorTile<OP, BM, N>(cb[k & 1], 0x100000);                          // vec2(k): consume C[k]
        if (k + 1 < NTILES)
            CubeTile<BM, K, N, BK>(cb[(k + 1) & 1], ap[(k + 1) & 1], B);     // cube(k+1)
        if (k + 2 < NTILES)
            VecPreOp<OP, BM, K>(A + (k + 2) * BM * K, ap[(k + 2) & 1], 0x200000);  // vec1(k+2)
    }
}
"""

# CUBE-output shapes (mixed_vc, mixed_cvc): the mirror of the vector-output shapes -- the cube is
# the OUTPUT (final) stage, so the bottleneck AIC's prologue idle behaves differently. VecCopyOp is
# the fp32 read-src -> write-dst vector (the middle stage of c->v->c); MixedVC / MixedCVC skew them.
_VC_CVC_KERNEL = r"""
// fp32 read-src -> write-dst vector op (C1' = C1 + C1 over [BM,N]); the middle stage of c->v->c.
// Unlike VectorTile (in-place) it writes a SEPARATE buffer, so cube1's out and cube2's in differ.
template <int OP, int BM, int N>
AICORE inline void VecCopyOp(__gm__ float *src, __gm__ float *dst, std::size_t ub_off) {
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT        = pto::Tile<pto::TileType::Vec, float, BM, N, pto::BLayout::RowMajor, BM, N>;
    VT v;
    TASSIGN(v, ub_off);
    ShapeDyn  shape(1, 1, 1, BM, N);
    StrideDyn stride(BM * N, BM * N, BM * N, N, 1);
    Global gs(src, shape, stride);
    Global gd(dst, shape, stride);
    TLOAD(v, gs);                                 // read C1 (GM->UB)
    if constexpr (OP == 0) { TADD(v, v, v); }     // C1' = C1 + C1
    else if constexpr (OP == 3) { TEXP(v, v); }
    TSTORE(gd, v);                                // write C1' (UB->GM): cube2 reads this next
}

// v -> c (prologue-only, CUBE output): vec1 pre-op feeds the cube, which STORES C as the output
// (no vec2). Skewed ping-pong on A'. Mirror of mixed_overlap: the CONSUMER (cube) is the bottleneck,
// so it IDLES during vec1(0) -> the fill = vec1_tile ADDS (total = max(cube,vec1) + vec1_tile).
template <int OP, int BM, int K, int N, int NTILES, int BK = (K < 128 ? K : 128)>
AICORE inline void MixedVC(__gm__ half *A, __gm__ half *B, __gm__ half *ap0, __gm__ half *ap1,
                           __gm__ float *cout) {
    __gm__ half *ap[2] = {ap0, ap1};
    VecPreOp<OP, BM, K>(A, ap[0], 0x200000);                                     // vec1(0) -> ap[0]
    for (int k = 0; k < NTILES; ++k) {
        if (k + 1 < NTILES)
            VecPreOp<OP, BM, K>(A + (k + 1) * BM * K, ap[(k + 1) & 1], 0x200000);  // vec1(k+1)
        CubeTile<BM, K, N, BK>(cout + k * BM * N, ap[k & 1], B);                   // cube(k): ap[k] -> output
    }
}

// c -> v -> c (flash-decode shape, CUBE output): cube1(A@B) -> vec(C1+C1) -> cube2(C1'@D) -> output.
// The AIC does BOTH cubes (cube_stage = cube1+cube2); the AIV does the middle vec. The two cubes'
// buffers are SEPARATE ping-pong pairs (c1[] for cube1->vec, cp[] for vec->cube2) so cube1[k+1] can
// pipeline while vec[k]/cube2[k-1] proceed -- upstream #1900's depth-2 target. cube2 reads cp
// reinterpreted as half (cost-model is data-agnostic; only the handoff ADDRESS drives the dep).
template <int OP, int BM, int K, int N, int NTILES, int BK1 = (K < 128 ? K : 128),
          int BK2 = (N < 128 ? N : 128)>
AICORE inline void MixedCVC(__gm__ half *A, __gm__ half *B, __gm__ half *D, __gm__ float *c1_0,
                            __gm__ float *c1_1, __gm__ float *cp0, __gm__ float *cp1, __gm__ float *cout) {
    __gm__ float *c1[2] = {c1_0, c1_1};
    __gm__ float *cp[2] = {cp0, cp1};
    CubeTile<BM, K, N, BK1>(c1[0], A, B);                                     // cube1(0): A@B -> c1[0]
    VecCopyOp<OP, BM, N>(c1[0], cp[0], 0x100000);                            // vec(0): c1[0] -> cp[0]
    if (NTILES > 1) CubeTile<BM, K, N, BK1>(c1[1], A + BM * K, B);            // cube1(1) -> c1[1]
    for (int k = 0; k < NTILES; ++k) {
        // cube2(k): C1' @ D -> output. C1' (fp32 in cp[]) reinterpreted as the cube's half operand.
        CubeTile<BM, N, N, BK2>(cout + k * BM * N, reinterpret_cast<__gm__ half *>(cp[k & 1]), D);
        if (k + 1 < NTILES)
            VecCopyOp<OP, BM, N>(c1[(k + 1) & 1], cp[(k + 1) & 1], 0x100000);            // vec(k+1)
        if (k + 2 < NTILES)
            CubeTile<BM, K, N, BK1>(c1[(k + 2) & 1], A + (k + 2) * BM * K, B);           // cube1(k+2)
    }
}
"""

# Distinct, widely-spaced fake GM bases so A / B / handoff buffers never alias in the dep tracker
# (addresses are never dereferenced -- the cost model is data-agnostic).
_GM_A = "reinterpret_cast<__gm__ half  *>(0x10000000)"
_GM_B = "reinterpret_cast<__gm__ half  *>(0x20000000)"
_GM_D = "reinterpret_cast<__gm__ half  *>(0x28000000)"   # cvc second matmul seed
_GM_C0 = "reinterpret_cast<__gm__ float *>(0x30000000)"
_GM_C1 = "reinterpret_cast<__gm__ float *>(0x40000000)"
# vcv / vc / cvc handoff buffers: A'-ping-pong (half) + C-ping-pong (float), widely spaced.
_GM_AP0 = "reinterpret_cast<__gm__ half  *>(0x50000000)"
_GM_AP1 = "reinterpret_cast<__gm__ half  *>(0x54000000)"
_GM_CB0 = "reinterpret_cast<__gm__ float *>(0x60000000)"
_GM_CB1 = "reinterpret_cast<__gm__ float *>(0x64000000)"

# (bm, K, N) tile shapes. Small so builds/runs are fast: fp16 in, fp32 acc, single K=128 step.
_SHAPES = [(128, 128, 128), (64, 128, 128)]
_NTILES = [1, 2, 4, 8]
_OP_ADD = 0  # C = C + C


def _overlap_fid(fid, op, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    static __gm__ half  *const A  = {_GM_A};\n"
        f"    static __gm__ half  *const B  = {_GM_B};\n"
        f"    static __gm__ float *const b0 = {_GM_C0};\n"
        f"    static __gm__ float *const b1 = {_GM_C1};\n"
        f"    MixedOverlap<{op}, {bm}, {k}, {n}, {nt}>(A, B, b0, b1);\n"
        f"}}"
    )


def _serial_fid(fid, op, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    static __gm__ half  *const A = {_GM_A};\n"
        f"    static __gm__ half  *const B = {_GM_B};\n"
        f"    static __gm__ float *const H = {_GM_C0};\n"
        f"    MixedSerial<{op}, {bm}, {k}, {n}, {nt}>(A, B, H);\n"
        f"}}"
    )


def gen_overlap():
    """mixed_overlap: skewed ping-pong producer. EXPECT total ~= max(cube,vec) + one tile fill,
    so overlap_factor -> ~1 as NTILES grows (fill amortizes) and -> 0 at NTILES=1.
    """
    defs, fids, idx = [_MIXED_KERNEL], [], []
    for bm, k, n in _SHAPES:
        for nt in _NTILES:
            fid = f"mo_bm{bm}_n{n}_k{k}_t{nt}"
            defs.append(_overlap_fid(fid, _OP_ADD, bm, k, n, nt))
            fids.append(fid)
            idx.append(dict(bm=bm, K=k, N=n, ntiles=nt, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_overlap", defs, fids, "MixedOverlap")
    return _save_index("mixed_overlap", dict(sweep=idx))


def gen_serial():
    """mixed_serial: single handoff buffer + B-operand chain. EXPECT total ~= cube + vec (sum),
    so overlap_factor ~= 0 for every NTILES (AIV active_start ~ AIC active_end each tile).
    """
    defs, fids, idx = [_MIXED_KERNEL], [], []
    for bm, k, n in _SHAPES:
        for nt in _NTILES:
            fid = f"ms_bm{bm}_n{n}_k{k}_t{nt}"
            defs.append(_serial_fid(fid, _OP_ADD, bm, k, n, nt))
            fids.append(fid)
            idx.append(dict(bm=bm, K=k, N=n, ntiles=nt, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_serial", defs, fids, "MixedSerial")
    return _save_index("mixed_serial", dict(sweep=idx))


# ── mixed_ddr_bound: sweep the AIC bottleneck across the GM<->compute boundary ──
# Reuses the SKEWED MixedOverlap kernel at fixed bm=128, NT=8 (fill amortized), sweeping K. The
# cube MAD (cube pipe) ~ bm*N*K grows with K; the fixp store (L0C->GM) ~ bm*N is CONSTANT in K;
# the mte2_aic reload (GM->L1, A+B per tile) ~ K*(bm+N) grows with K. So small K is store/GM-bound
# (fixp dominates the AIC) and large K shifts the AIC's dominant pipe toward MAD/reload. The point:
# the per-unit ACTIVE wall already SUBSUMES all the GM ports (it is the overlapped critical path
# through them), so total = max(cube_stage, vec_stage) + fill holds across the whole sweep with NO
# separate `ddr` max term. C=C+C keeps the AIV stage cheap-compute -> GM-bound (load+store).
_DDR_NTILES = 8
_DDR_BM = 128
_DDR_K = [16, 32, 64, 128, 256, 512]
_DDR_N = [128, 256]


def gen_ddr_bound():
    """mixed_ddr_bound: skewed kernel, K sweep at bm=128/NT=8 (+ N in {128,256}). Show the AIC
    dominant pipe shift (gm/store -> reload/MAD) while total stays = max(stage)+fill (ddr subsumed).
    """
    defs, fids, idx = [_MIXED_KERNEL], [], []
    for n in _DDR_N:
        for k in _DDR_K:
            fid = f"md_n{n}_k{k}"
            defs.append(_overlap_fid(fid, _OP_ADD, _DDR_BM, k, n, _DDR_NTILES))
            fids.append(fid)
            idx.append(dict(bm=_DDR_BM, K=k, N=n, ntiles=_DDR_NTILES, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_ddr_bound", defs, fids, "MixedDdrBound")
    return _save_index("mixed_ddr_bound", dict(sweep=idx))


# ── mixed_contention: cross-unit shared-HBM-read contention (the mlsys26 par()/ddr_lat term) ──
# The perf-sim pools the cube read (GM_TO_L1 / mte2_aic) AND the vector read (GM_TO_UB / mte2_aiv)
# onto ONE `total_read_gibs` knob (arch_config.hpp HillBandwidthModel::GroupTotal returns the same
# value for both GM read pipes). BwEff caps EACH read pipe at min(peak, total_read/ncores), with
# ncores = block_dim (SetActiveCoreCount in LAUNCH_KERNEL). So as block_dim grows past a pipe's knee
# (900/peak), that pipe's BW drops to 900/B and its mte2 cycles inflate ~B. We run the SKEWED mixed
# kernel multi-core, uncapped (total_read=0, the flat no-contention baseline) vs capped (900), and
# read mte2_aic + mte2_aiv to confirm BOTH throttle to the same 900/B from the one shared pool.
#
# The cap is set IN-KERNEL (MakeFlatHillModel keeps flat peaks 135/100.9 + adds the read cap). Note
# PTO_BW_MODE=fitted leaves total_read at 0 (no cap) AND would be overridden by this in-kernel set,
# so the contention is grounded here in the default (flat) binary, not via the env. Config: small K
# (cheap MAD) keeps the kernel GM-read-bound so the shared read pool is the binding resource.
_CT_BM, _CT_K, _CT_N, _CT_NT = 128, 32, 128, 4
_CT_BS = [1, 2, 4, 8, 16, 24]


def _contention_fid(fid, cap, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    auto _m = pto::mocker::evaluator::MakeFlatHillModel();\n"
        f"    _m.total_read_gibs = {cap};          // 0 => uncapped (flat baseline); 900 => shared pool\n"
        f"    pto::mocker::evaluator::SetHillBandwidthModel(_m);\n"
        f"    static __gm__ half  *const A  = {_GM_A};\n"
        f"    static __gm__ half  *const B  = {_GM_B};\n"
        f"    static __gm__ float *const b0 = {_GM_C0};\n"
        f"    static __gm__ float *const b1 = {_GM_C1};\n"
        f"    MixedOverlap<{_OP_ADD}, {bm}, {k}, {n}, {nt}>(A, B, b0, b1);\n"
        f"}}"
    )


def gen_contention():
    """mixed_contention: multi-core skewed kernel, uncapped vs total_read=900 capped, B sweep.
    EXPECT uncapped per-core mte2 constant in B; capped cube reads throttle at B~7 (900/135) and
    vector reads at B~9 (900/100.9), BOTH collapsing to 900/B -> one shared cross-unit read pool.
    """
    hbm = C.HBM_AGGREGATE_GIBS
    defs, fids, cfgs, sweep = [_MIXED_KERNEL], [], {}, []
    for B in _CT_BS:
        un, cap = f"mxc_un_{B}", f"mxc_cap_{B}"
        defs.append(_contention_fid(un, "0.0", _CT_BM, _CT_K, _CT_N, _CT_NT))
        defs.append(_contention_fid(cap, f"{hbm}", _CT_BM, _CT_K, _CT_N, _CT_NT))
        fids += [un, cap]
        cfgs[un] = f"({B}, nullptr, nullptr)"
        cfgs[cap] = f"({B}, nullptr, nullptr)"
        sweep.append(dict(B=B, un=un, cap=cap))
    C.write_testcase("mixed_contention", defs, fids, "MixedContention", launch_cfgs=cfgs)
    return _save_index("mixed_contention",
                       dict(hbm=hbm, bm=_CT_BM, K=_CT_K, N=_CT_N, nt=_CT_NT, sweep=sweep))


# ── mixed_vcv: 3-stage vec1 -> cube -> vec2, to ground how the pipeline FILL deepens ──
# The 2-stage kernels can only show fill = one cube tile. Prepending a cheap vector PRE-op (vec1:
# A' = A + A on the cube's left operand) makes the deepest consumer (vec2) unable to start until
# vec1(0) AND cube(0) have both landed -> fill = vec1_tile + cube_tile (a 2-stage prologue). vec1 and
# vec2 BOTH pool on the AIV, so vec_stage = vec1_total + vec2_total. Skewed/ping-ponged so the 3
# stages overlap. Same shape as mixed_overlap (bm=128,N=128,K=128) for a direct fill comparison.
_VCV_SHAPE = (128, 128, 128)  # bm, K, N
_VCV_NTILES = [1, 2, 4, 8]


def _vcv_fid(fid, op, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    static __gm__ half  *const A   = {_GM_A};\n"
        f"    static __gm__ half  *const B   = {_GM_B};\n"
        f"    static __gm__ half  *const ap0 = {_GM_AP0};\n"
        f"    static __gm__ half  *const ap1 = {_GM_AP1};\n"
        f"    static __gm__ float *const c0  = {_GM_CB0};\n"
        f"    static __gm__ float *const c1  = {_GM_CB1};\n"
        f"    MixedVCV<{op}, {bm}, {k}, {n}, {nt}>(A, B, ap0, ap1, c0, c1);\n"
        f"}}"
    )


def gen_vcv():
    """mixed_vcv: 3-stage skewed vec1->cube->vec2. EXPECT total ~= max(cube, vec1+vec2) + fill with
    fill = vec1_tile + cube_tile -- a DEEPER prologue than the 2-stage mixed_overlap's one cube tile.
    """
    bm, k, n = _VCV_SHAPE
    defs, fids, idx = [_MIXED_KERNEL, _VCV_KERNEL], [], []
    for nt in _VCV_NTILES:
        fid = f"vcv_bm{bm}_n{n}_k{k}_t{nt}"
        defs.append(_vcv_fid(fid, _OP_ADD, bm, k, n, nt))
        fids.append(fid)
        idx.append(dict(bm=bm, K=k, N=n, ntiles=nt, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_vcv", defs, fids, "MixedVcv")
    return _save_index("mixed_vcv", dict(sweep=idx))


# ── mixed_vc / mixed_cvc: CUBE-output shapes (AIC is the final/output stage) ──
# vc  = v -> c  (prologue-only): the cube CONSUMES vec1 and stores C. The cube (bottleneck) idles
#       during vec1(0), so the fill = vec1_tile ADDS -- the mirror of mixed_overlap's cube_tile fill.
# cvc = c -> v -> c: the AIC runs BOTH cubes; the two cubes must pipeline (separate ping-pong pairs)
#       or they serialize (#1900). cube1 starts at t=0 on the bottleneck AIC, so the fill is absorbed.
_VC_SHAPE = (128, 128, 128)  # bm, K, N
_VC_NTILES = [1, 2, 4, 8]


def _vc_fid(fid, op, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    static __gm__ half  *const A   = {_GM_A};\n"
        f"    static __gm__ half  *const B   = {_GM_B};\n"
        f"    static __gm__ half  *const ap0 = {_GM_AP0};\n"
        f"    static __gm__ half  *const ap1 = {_GM_AP1};\n"
        f"    static __gm__ float *const co  = {_GM_C0};\n"
        f"    MixedVC<{op}, {bm}, {k}, {n}, {nt}>(A, B, ap0, ap1, co);\n"
        f"}}"
    )


def _cvc_fid(fid, op, bm, k, n, nt):
    return (
        f"void {fid}() {{\n"
        f"    static __gm__ half  *const A   = {_GM_A};\n"
        f"    static __gm__ half  *const B   = {_GM_B};\n"
        f"    static __gm__ half  *const D   = {_GM_D};\n"
        f"    static __gm__ float *const c10 = {_GM_AP0.replace('half  *', 'float *')};\n"
        f"    static __gm__ float *const c11 = {_GM_AP1.replace('half  *', 'float *')};\n"
        f"    static __gm__ float *const cp0 = {_GM_CB0};\n"
        f"    static __gm__ float *const cp1 = {_GM_CB1};\n"
        f"    static __gm__ float *const co  = {_GM_C0};\n"
        f"    MixedCVC<{op}, {bm}, {k}, {n}, {nt}>(A, B, D, c10, c11, cp0, cp1, co);\n"
        f"}}"
    )


def gen_vc():
    """mixed_vc: v->c prologue-only, CUBE output. EXPECT the cube is the bottleneck and idles during
    vec1(0) -> fill = vec1_tile ADDS: total ~= max(cube_stage, vec1_stage) + vec1_tile."""
    bm, k, n = _VC_SHAPE
    defs, fids, idx = [_MIXED_KERNEL, _VCV_KERNEL, _VC_CVC_KERNEL], [], []
    for nt in _VC_NTILES:
        fid = f"vc_bm{bm}_n{n}_k{k}_t{nt}"
        defs.append(_vc_fid(fid, _OP_ADD, bm, k, n, nt))
        fids.append(fid)
        idx.append(dict(bm=bm, K=k, N=n, ntiles=nt, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_vc", defs, fids, "MixedVc")
    return _save_index("mixed_vc", dict(sweep=idx))


def gen_cvc():
    """mixed_cvc: c->v->c, CUBE output. EXPECT the AIC (cube1+cube2) is the bottleneck; the two cubes
    PIPELINE (total ~= max(cube1+cube2, vec) + fill, not the cube1+vec+cube2 serial sum); fill absorbed."""
    bm, k, n = _VC_SHAPE
    defs, fids, idx = [_MIXED_KERNEL, _VCV_KERNEL, _VC_CVC_KERNEL], [], []
    for nt in _VC_NTILES:
        fid = f"cvc_bm{bm}_n{n}_k{k}_t{nt}"
        defs.append(_cvc_fid(fid, _OP_ADD, bm, k, n, nt))
        fids.append(fid)
        idx.append(dict(bm=bm, K=k, N=n, ntiles=nt, op=_OP_ADD, fid=fid))
    C.write_testcase("mixed_cvc", defs, fids, "MixedCvc")
    return _save_index("mixed_cvc", dict(sweep=idx))


ALL = {
    "mixed_overlap": gen_overlap,
    "mixed_serial": gen_serial,
    "mixed_ddr_bound": gen_ddr_bound,
    "mixed_contention": gen_contention,
    "mixed_vcv": gen_vcv,
    "mixed_vc": gen_vc,
    "mixed_cvc": gen_cvc,
}
