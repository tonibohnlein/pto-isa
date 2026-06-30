# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Experiment definitions for the VECTOR-tile cost-model study. Each generator writes a
# perf-sim testcase (testcase/<name>/{main.cpp,CMakeLists.txt}) and returns an index used
# by analyze.py. Kernels are self-contained in main.cpp (no external kernel header).
# NOTE: testcase names here must also be listed in testcase/CMakeLists.txt (ALL_TESTCASES).

import json

import common as C


def _save_index(name, index):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / f"{name}_index.json").write_text(json.dumps(index, indent=2))
    return index


# A homogeneous chain of NOPS pointwise vector ops over a [ROWS,COLS] Vec (UB) tile. The
# dst tile is Vec -> the ops charge PipeKey::VECTOR (vec_cycles). Keeping COLS a multiple
# of 256/sizeof(T) (64 fp32) takes the fast path: one fused call, repeat = ROWS*COLS/epr.
# OP selects add/mul/div/exp; back-to-back ops share ONE stream so head+tail is paid once.
_VEC_CHAIN_KERNEL = r"""
template <int OP, typename T, int ROWS, int COLS, int NOPS>
AICORE inline void VecChain(__gm__ T *src) {
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT        = pto::Tile<pto::TileType::Vec, T, ROWS, COLS, pto::BLayout::RowMajor, ROWS, COLS>;
    constexpr int nbytes = ROWS * COLS * (int)sizeof(T);
    VT a, b, c;
    TASSIGN(a, 0);
    TASSIGN(b, nbytes);
    TASSIGN(c, 2 * nbytes);
    ShapeDyn  shape(1, 1, 1, ROWS, COLS);
    StrideDyn stride(ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1);
    Global g(src, shape, stride);
    TLOAD(a, g);
    TLOAD(b, g);
    for (int i = 0; i < NOPS; ++i) {
        if constexpr (OP == 0) { TADD(c, a, b); }
        else if constexpr (OP == 1) { TMUL(c, a, b); }
        else if constexpr (OP == 2) { TDIV(c, a, b); }
        else if constexpr (OP == 3) { TEXP(c, a); }
    }
}
"""

_EPR = C.VEC_REG_BYTES // 4   # fp32 elements per repeat (64)


def gen_pointwise():
    """vec_pointwise: ground the per-op VECTOR formula slope*repeat + once-per-stream head/tail.

    (A) slope sweep -- per op, sweep repeat (COLS=64*repeat, ROWS=1, NOPS=1): back out slope
        and the startup intercept, compare to the device-calibrated stub.
    (B) chain sweep -- op=add, fixed repeat, sweep NOPS: show vec_cycles = head+tail + NOPS*
        slope*repeat (startup paid ONCE), vs mlsys26's per-op NOPS*(head+slope*repeat+tail).
    """
    repeats = [1, 2, 4, 8, 16, 32, 64]
    chain_repeat = 8
    chain_nops = [1, 2, 4, 8, 16]
    defs, fids = [_VEC_CHAIN_KERNEL], []
    slope_idx, chain_idx = [], []

    def emit(fid, op, rows, cols, nops):
        defs.append(f"void {fid}() {{ VecChain<{op}, float, {rows}, {cols}, {nops}>(nullptr); }}")
        fids.append(fid)

    # (A) slope sweep: each op, single op, sweep repeat
    for op, meta in C.VEC_OPS.items():
        for r in repeats:
            fid = f"vp_{meta['name']}_r{r}"
            emit(fid, op, 1, _EPR * r, 1)
            slope_idx.append(dict(op=op, name=meta["name"], repeat=r, cols=_EPR * r, fid=fid))
    # (B) chain sweep: add, fixed repeat, sweep chain length
    for n in chain_nops:
        fid = f"vp_chain_n{n}"
        emit(fid, 0, 1, _EPR * chain_repeat, n)
        chain_idx.append(dict(nops=n, repeat=chain_repeat, fid=fid))

    C.write_testcase("vec_pointwise", defs, fids, "VecPointwise")
    return _save_index("pointwise", dict(slope=slope_idx, chain=chain_idx,
                                         chain_op=0, chain_repeat=chain_repeat))


# Reductions. TROWSUM ([H,W]->[H,1], reduce W) lowers to a tree of count-mode vadd passes +
# a final vcadd, EACH separated by pipe_barrier(PIPE_V) -- which resets the VEC queue, so every
# pass re-pays head+tail. Count mode forces repeat=0 -> cost is ROWS-independent, linear in
# COLS/64. TCOLSUM ([H,W]->[1,W], reduce H, binary) is a pairwise vadd tree across rows.
_VEC_REDUCE_KERNEL = r"""
// Mirror the ST trowsum/tcolsum tests: full [ROWS,COLS] template (aligned Cols) with
// DYNAMIC valid dims (-1,-1) + runtime ctor giving each tile its real valid shape. No
// TLOAD -- the cost model is data-agnostic, so TASSIGN + the reduce op is enough.
template <typename T, int ROWS, int COLS>
AICORE inline void RowSum() {
    using TD = pto::Tile<pto::TileType::Vec, T, ROWS, COLS, pto::BLayout::RowMajor, -1, -1>;
    constexpr int nbytes = ROWS * COLS * (int)sizeof(T);
    TD src(ROWS, COLS), tmp(ROWS, COLS), dst(ROWS, COLS);
    TASSIGN(src, 0);
    TASSIGN(tmp, nbytes);
    TASSIGN(dst, 2 * nbytes);
    TROWSUM(dst, src, tmp);          // [ROWS,COLS] -> [ROWS,1] (reduce W)
}
template <typename T, int ROWS, int COLS>
AICORE inline void ColSumBin() {
    using TD = pto::Tile<pto::TileType::Vec, T, ROWS, COLS, pto::BLayout::RowMajor, -1, -1>;
    constexpr int nbytes = ROWS * COLS * (int)sizeof(T);
    TD src(ROWS, COLS), tmp((ROWS / 2 > 0 ? ROWS / 2 : 1), COLS), dst(1, COLS);
    TASSIGN(src, 0);
    TASSIGN(tmp, nbytes);
    TASSIGN(dst, 2 * nbytes);
    TCOLSUM(dst, src, tmp, true);    // [ROWS,COLS] -> [1,COLS] (reduce H, binary tree)
}
"""


def gen_reduce():
    """vec_reduce: ground the reduction cost vs mlsys26's lumped slope_reduce*repeat.

    (A) TROWSUM COLS sweep (ROWS fixed): cost ~ linear in COLS/64 (tree depth).
    (B) TROWSUM ROWS sweep (COLS fixed): cost ~ CONSTANT (count-mode -> ROWS-independent),
        the headline gap vs mlsys26's repeat = ROWS*COLS.
    (C) TCOLSUM binary ROWS sweep: pairwise vadd tree across rows.
    """
    defs, fids = [_VEC_REDUCE_KERNEL], []
    rs_cols, rs_rows = [], []
    cs_rows = []

    def emit(fid, call):
        defs.append(f"void {fid}() {{ {call} }}")
        fids.append(fid)

    # (A) TROWSUM: fixed ROWS=8, sweep the reduced dim COLS (UB: 3 x ROWS*COLS*4 bytes)
    for cols in [64, 128, 256, 512, 1024]:
        fid = f"rs_c{cols}"
        emit(fid, f"RowSum<float, 8, {cols}>();")
        rs_cols.append(dict(rows=8, cols=cols, fid=fid))
    # (B) TROWSUM: fixed COLS=128, sweep ROWS -> should be FLAT (count-mode, ROWS-independent)
    for rows in [8, 16, 32, 64]:
        fid = f"rs_r{rows}"
        emit(fid, f"RowSum<float, {rows}, 128>();")
        rs_rows.append(dict(rows=rows, cols=128, fid=fid))
    # (C) TCOLSUM binary: fixed COLS=128, sweep the reduced dim ROWS (powers of 2)
    for rows in [2, 4, 8, 16, 32, 64]:
        fid = f"cs_r{rows}"
        emit(fid, f"ColSumBin<float, {rows}, 128>();")
        cs_rows.append(dict(rows=rows, cols=128, fid=fid))

    C.write_testcase("vec_reduce", defs, fids, "VecReduce")
    return _save_index("reduce", dict(rs_cols=rs_cols, rs_rows=rs_rows, cs_rows=cs_rows))


# UB-overflow streaming. A softmax chain (rowmax -> sub -> exp -> rowsum -> div) either fits
# UB (MATERIALIZED, one pass) or streams the reduced axis in chunks. The canonical pto-isa
# emit is ONLINE/flash: each chunk's exp runs ONCE; a thin [H,1] correction (max-merge +
# running-sum rescale) per chunk replaces any re-read. So the wide-body recompute factor is
# ~1, NOT mlsys26's #reductions+1. We measure the streamed/materialized vec_cycles ratio.
_VEC_STREAM_KERNEL = r"""
// Online softmax numerator per chunk: rowmax -> center -> exp -> rowsum over [ROWS,CW]. The
// exp runs ONCE per element regardless of NCHUNKS (the wide body splits, it does not repeat).
// The full flash emit adds a thin [H,1] max-merge + running-sum rescale per chunk (a few
// ColMajor/RowMajor reduce-tile ops) -- a small surcharge we account for analytically, omitted
// here to keep one clean RowMajor wide-body kernel. NCHUNKS=1 is the materialized single pass.
template <typename T, int ROWS, int COLS, int NCHUNKS>
AICORE inline void SoftmaxStream() {
    constexpr int CW = COLS / NCHUNKS;                 // chunk width (a multiple of 64)
    using CT = pto::Tile<pto::TileType::Vec, T, ROWS, CW, pto::BLayout::RowMajor, ROWS, CW>;
    using RT = pto::Tile<pto::TileType::Vec, T, ROWS, 1, pto::BLayout::ColMajor, ROWS, 1>;
    constexpr int cb = ROWS * CW * (int)sizeof(T);
    CT xc, tmp; RT mx, sm;
    TASSIGN(xc, 0);
    TASSIGN(tmp, cb);
    TASSIGN(mx, 2 * cb);
    TASSIGN(sm, 2 * cb + ROWS * (int)sizeof(T));
    for (int j = 0; j < NCHUNKS; ++j) {
        TROWMAX(mx, xc, tmp);          // [H,CW] -> [H,1]  chunk max
        TROWEXPANDSUB(xc, xc, mx);     // center (broadcast [H,1] over CW)
        TEXP(xc, xc);                  // exp ONCE over the chunk
        TROWSUM(sm, xc, tmp);          // [H,CW] -> [H,1]  chunk sum
    }
}
"""


def gen_stream():
    """vec_stream: ground mlsys26's UB-overflow recompute multiplier (N_passes=#reductions+1).

    A softmax chain over [ROWS,COLS] streamed over COLS in NCHUNKS chunks (online/flash). The
    wide-body exp runs once per element regardless of NCHUNKS, so the streamed/materialized
    vec_cycles ratio stays ~1 + O(NCHUNKS) thin corrections -- NOT mlsys26's flat 3x.
    """
    ROWS, COLS = 8, 512
    nchunks = [1, 2, 4, 8]
    defs, fids, idx = [_VEC_STREAM_KERNEL], [], []
    for n in nchunks:
        fid = f"sm_n{n}"
        defs.append(f"void {fid}() {{ SoftmaxStream<float, {ROWS}, {COLS}, {n}>(); }}")
        fids.append(fid)
        idx.append(dict(nchunks=n, rows=ROWS, cols=COLS, fid=fid))
    C.write_testcase("vec_stream", defs, fids, "VecStream")
    return _save_index("stream", dict(rows=ROWS, cols=COLS, sweep=idx))


# Reduced-axis cross-core split (the vector analog of the cube split-K). A sink reduction
# over [H,W] split S ways: each core reduces its band [H, Wc=W/S] -> [H,1] and writes that
# thin partial to GM (the S partials atomic-add merge). Mirroring gml1_splitk, we measure the
# PER-CORE subproblem on one core: vec (the reduce) should drop ~Wc, and mte3 (the [H,1]
# partial store) is a constant floor -- the merge cost is then S * that thin store.
_VEC_SPLITS_KERNEL = r"""
template <typename T, int H, int WC>
AICORE inline void SplitReduce(__gm__ T *out) {
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT = pto::Tile<pto::TileType::Vec, T, H, WC, pto::BLayout::RowMajor, H, WC>;
    using RT = pto::Tile<pto::TileType::Vec, T, H, 1, pto::BLayout::ColMajor, H, 1>;
    constexpr int nb = H * WC * (int)sizeof(T);
    VT src, tmp; RT partial;
    TASSIGN(src, 0);
    TASSIGN(tmp, nb);
    TASSIGN(partial, 2 * nb);
    TROWSUM(partial, src, tmp);                  // per-core band reduce [H,WC] -> [H,1]
    ShapeDyn  shape(1, 1, 1, H, 1);
    StrideDyn stride(H, H, H, 1, 1);
    Global g(out, shape, stride);
    TSTORE(g, partial);                          // write the thin [H,1] partial (UB->GM / mte3)
}
"""


def gen_splitS():
    """vec_splitS: the per-core subproblem of a sink-reduction split S ways (cube split-K analog).

    A reduction over [H,W] split S ways: per core reduces [H, Wc=W/S] + stores the [H,1]
    partial. vec (the reduce) drops ~Wc (parallelizes); mte3 (the partial store) is a const
    floor; the cross-core merge is S * that thin store -- the compute(~1/S) vs merge(~S) tradeoff.
    """
    # W=512 (not the H*W==8192 shapes 8x1024/16x512/... that trigger the vcgadd FP32-reduce
    # fast path) so every per-core band [8, Wc] takes the generic tree and matches the formula.
    H, W = 8, 512
    defs, fids, idx = [_VEC_SPLITS_KERNEL], [], []
    for S in [1, 2, 4, 8]:
        wc = W // S
        fid = f"sp_s{S}"
        defs.append(f"void {fid}() {{ SplitReduce<float, {H}, {wc}>(nullptr); }}")
        fids.append(fid)
        idx.append(dict(S=S, wc=wc, H=H, W=W, fid=fid))
    C.write_testcase("vec_splitS", defs, fids, "VecSplitS")
    return _save_index("splitS", dict(H=H, W=W, sweep=idx))


ALL = {
    "vec_pointwise": gen_pointwise,
    "vec_reduce": gen_reduce,
    "vec_stream": gen_stream,
    "vec_splitS": gen_splitS,
}
