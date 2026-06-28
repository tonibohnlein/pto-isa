# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Experiment definitions for the L0-tile cost-model study. Each generator writes a
# perf-sim testcase (testcase/<name>/{main.cpp,CMakeLists.txt}) and returns an index
# of (kernel-fn -> config) used by analyze.py. See README.md.

import json

import common as C

L0A, L0B, L0C, PING = C.L0A, C.L0B, C.L0C, C.L0_PING


def _save_index(name, index):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / f"{name}_index.json").write_text(json.dumps(index, indent=2))
    return index


# ----------------------------------------------------------------------------- sweep
# Split-K aspect/k sweep over the gemm_performance reference kernel. Validates the
# CUBE closed form and the L0A/L0B asymmetry (tall tiles beat their transpose).
def gen_sweep():
    A0 = B0 = L0A // (2 * 2)   # bf16, operands double-buffered
    C0 = L0C // 4              # fp32 acc, single L0C
    problems = [  # (label, M, N, K, k-list, min_area)
        ("A_512x128x512", 512, 128, 512, [16, 32, 64, 128], 8192),
        ("B_512x512x512", 512, 512, 512, [32, 64], 16384),
    ]
    defs, fids, index = [], [], []
    n = 0
    for (label, M, N, K, ks, min_area) in problems:
        for k in ks:
            if K % k:
                continue
            for m in C.divisors(M):
                for nn in C.divisors(N):
                    if m * k <= A0 and k * nn <= B0 and m * nn <= C0 and m * nn >= min_area:
                        fid = f"sw{n}"
                        defs.append(C.emit_split_k_baseline(fid, M, K, N, m, k, nn))
                        fids.append(fid)
                        index.append(dict(id=fid, label=label, M=M, N=N, K=K, m=m, k=k, n=nn))
                        n += 1
    C.write_testcase("gemm_sweep", "gemm_performance_kernel.cpp", defs, fids, "GemmSweep")
    return _save_index("sweep", index)


# ----------------------------------------------------------------------------- fullk
# full-K operand reuse: no-reuse baseline vs A-stationary vs B-stationary. Validates
# the reuse saving AND the bandwidth-weighted stationary choice (vs bytes-only).
def gen_fullk():
    problems = [  # (M, N, K, [(baseM, baseN), ...])
        (512, 512, 64, [(128, 128), (256, 128), (128, 256), (256, 64), (64, 256), (128, 64), (64, 128)]),
        (512, 512, 128, [(128, 128), (128, 64), (64, 128)]),
    ]
    defs, fids, index = [], [], []
    n = 0
    for (M, N, K, tiles) in problems:
        for (bm, bn) in tiles:
            if not (bm * K * 2 <= PING and K * bn * 2 <= PING and M % bm == 0 and N % bn == 0):
                continue
            tag = f"{M}x{N}x{K}_{bm}x{bn}"
            roles = [
                ("base", C.emit_split_k_baseline(f"fk_base_{n}", M, K, N, bm, K, bn)),
                ("astat", f"void fk_astat_{n}() {{ RunGemmFullKReuse<float, half, half, "
                          f"{M}, {K}, {N}, {bm}, {bn}, true>(nullptr, nullptr, nullptr); }}"),
                ("bstat", f"void fk_bstat_{n}() {{ RunGemmFullKReuse<float, half, half, "
                          f"{M}, {K}, {N}, {bm}, {bn}, false>(nullptr, nullptr, nullptr); }}"),
            ]
            for role, d in roles:
                fid = f"fk_{role}_{n}"
                defs.append(d)
                fids.append(fid)
                index.append(dict(id=fid, role=role, tag=tag, M=M, N=N, K=K, baseM=bm, baseN=bn))
            n += 1
    C.write_testcase("gemm_fullk", "fullk_reuse_kernel.cpp", defs, fids, "GemmFullK")
    return _save_index("fullk", index)


# ------------------------------------------------------------------------------- dbc
# L0C double-buffering: best single-L0C tile vs same tile double-buffered vs bigger
# single tile. Validates that hiding the exposed FIXPIPE drain beats halving C0.
def gen_dbc():
    problems = [  # (M, N, K, baseK, big(m,n), small(m,n))
        (512, 512, 64, 64, (256, 128), (128, 128)),
        (512, 512, 128, 64, (256, 128), (128, 128)),
        (512, 512, 256, 64, (256, 128), (128, 128)),
    ]
    defs, fids, index = [], [], []
    n = 0
    for (M, N, K, bk, (bM, bN), (sM, sN)) in problems:
        tag = f"{M}x{N}x{K}"
        for (role, m, nn, numc) in [("big1", bM, bN, 1), ("small1", sM, sN, 1), ("small2", sM, sN, 2)]:
            assert numc * m * nn * 4 <= L0C
            fid = f"dbc_{role}_{n}"
            defs.append(f"void {fid}() {{ RunGemmSplitKDBC<float, half, half, "
                        f"{M}, {K}, {N}, {m}, {bk}, {nn}, {numc}>(nullptr, nullptr, nullptr); }}")
            fids.append(fid)
            index.append(dict(id=fid, role=role, tag=tag, M=M, N=N, K=K, m=m, baseK=bk, n=nn, numc=numc))
        n += 1
    C.write_testcase("gemm_dbc", "dbc_kernel.cpp", defs, fids, "GemmDBC")
    return _save_index("dbc", index)


# -------------------------------------------------------------------------- accblock
# Variant 3 -- accumulator/C-blocking: NACC L0C accumulators give split-K the A-reuse
# that otherwise needs full-K. MTE1 should drop as NACC grows (A extracts / NACC).
def gen_accblock():
    problems = [  # (M, N, K, baseK, baseM, baseN, [NACC, ...])
        (512, 512, 128, 64, 128, 64, [1, 2, 4]),
        (512, 512, 256, 64, 128, 64, [1, 2, 4]),
        (512, 512, 128, 64, 128, 128, [1, 2]),
    ]
    defs, fids, index = [], [], []
    n = 0
    for (M, N, K, bk, bm, bn, naccs) in problems:
        tag = f"{M}x{N}x{K}_{bm}x{bn}"
        for nacc in naccs:
            assert nacc * bm * bn * 4 <= L0C and N % (bn * nacc) == 0
            fid = f"ab_{n}_{nacc}"
            defs.append(f"void {fid}() {{ RunGemmAccBlock<float, half, half, "
                        f"{M}, {K}, {N}, {bm}, {bk}, {bn}, {nacc}>(nullptr, nullptr, nullptr); }}")
            fids.append(fid)
            index.append(dict(id=fid, tag=tag, M=M, N=N, K=K, baseM=bm, baseK=bk, baseN=bn, nacc=nacc))
        n += 1
    C.write_testcase("gemm_accblock", "accblock_kernel.cpp", defs, fids, "GemmAccBlock")
    return _save_index("accblock", index)


# --------------------------------------------------------------------------- asymbuf
# Variant 4 -- asymmetric-buffered full-K: stationary single-buffered (full L0), moving
# operand single (MOVDB=1) vs double (MOVDB=2). Same traffic; MOVDB=2 hides the B load.
def gen_asymbuf():
    problems = [  # (M, N, K, [(baseM, baseN), ...])
        (512, 512, 64, [(128, 128), (256, 128), (128, 256)]),
        (512, 512, 128, [(128, 128)]),
    ]
    defs, fids, index = [], [], []
    n = 0
    for (M, N, K, tiles) in problems:
        for (bm, bn) in tiles:
            # stationary A single-buffered (full L0A); moving B double-buffered needs 2 slots.
            if not (bm * K * 2 <= L0A and 2 * K * bn * 2 <= L0B and M % bm == 0 and N % bn == 0):
                continue
            tag = f"{M}x{N}x{K}_{bm}x{bn}"
            for movdb in [1, 2]:
                fid = f"as_{n}_{movdb}"
                defs.append(f"void {fid}() {{ RunGemmFullKAsymBuf<float, half, half, "
                            f"{M}, {K}, {N}, {bm}, {bn}, {movdb}>(nullptr, nullptr, nullptr); }}")
                fids.append(fid)
                index.append(dict(id=fid, tag=tag, M=M, N=N, K=K, baseM=bm, baseN=bn, movdb=movdb))
            n += 1
    C.write_testcase("gemm_asymbuf", "asymbuf_kernel.cpp", defs, fids, "GemmAsymBuf")
    return _save_index("asymbuf", index)


ALL = {
    "gemm_sweep": gen_sweep,
    "gemm_fullk": gen_fullk,
    "gemm_dbc": gen_dbc,
    "gemm_accblock": gen_accblock,
    "gemm_asymbuf": gen_asymbuf,
}
