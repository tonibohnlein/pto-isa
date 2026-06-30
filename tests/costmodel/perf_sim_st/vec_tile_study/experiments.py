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


ALL = {
    "vec_pointwise": gen_pointwise,
}
