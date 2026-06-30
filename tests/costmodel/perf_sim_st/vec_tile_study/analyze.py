# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Analysis for the VECTOR-tile cost-model study: reads the perf-sim CSVs and grounds each
# mechanism against pto-isa's device-calibrated stub. Run via run.py. See README.md.

import json

import common as C


def _index(name):
    return json.loads((C.RESULTS_DIR / f"{name}_index.json").read_text())


def _lsq(xs, ys):
    """Least-squares (slope, intercept) of ys vs xs."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")
    return slope, my - slope * mx


def analyze_pointwise():
    print("\n=== pointwise: VECTOR cycles == slope*repeat + once-per-stream (head+tail) ? ===")
    idx = _index("pointwise")

    # (A) per-op slope + startup intercept vs the device-calibrated stub
    print("  (A) slope sweep -- back out slope & startup from vec_cycles(repeat), NOPS=1")
    print(f"      {'op':>4} {'instr':>10} | {'slope':>6} {'cal':>4} | {'startup':>8} {'cal(h+t)':>9} | verdict")
    by_op = {}
    for x in idx["slope"]:
        by_op.setdefault(x["op"], []).append(x)
    for op, rows in sorted(by_op.items()):
        rows.sort(key=lambda r: r["repeat"])
        xs = [r["repeat"] for r in rows]
        ys = [C.read_aiv(r["fid"])["vec"] for r in rows]
        slope, intc = _lsq(xs, ys)
        cal = C.VEC_OPS[op]
        ok = abs(slope - cal["slope"]) < 0.3 and abs(intc - cal["ht"]) < 4
        print(f"      {cal['name']:>4} {cal['instr']:>10} | {slope:>6.2f} {cal['slope']:>4} | "
              f"{intc:>8.1f} {cal['ht']:>9} | {'OK' if ok else 'CHECK'}")

    # (B) chain length: once-per-stream startup (perf-sim) vs per-op startup (mlsys26)
    print("  (B) chain sweep -- op=add, repeat=8: startup paid ONCE (perf-sim) vs PER-OP (mlsys26)")
    crep = idx["chain_repeat"]
    print(f"      {'NOPS':>4} | {'sim_vec':>7} {'perfsim':>7} {'e%':>5} | {'mlsys26':>7} {'over':>5}")
    rows = sorted(idx["chain"], key=lambda r: r["nops"])
    for x in rows:
        n = x["nops"]
        sim = C.read_aiv(x["fid"])["vec"]
        pred = C.perfsim_chain_cycles(idx["chain_op"], crep, n)        # ht + n*slope*repeat
        mly = C.mlsys_chain_cycles(crep, n)                            # n*(head+slope*repeat+tail)
        err = (sim - pred) / pred * 100 if pred else float("nan")
        over = mly / sim if sim else float("nan")
        print(f"      {n:>4} | {sim:>7} {pred:>7.0f} {err:>+4.1f}% | {mly:>7.0f} {over:>4.1f}x")
    print("  -> the device stub pays head+tail ONCE per vector stream (back-to-back ops overlap")
    print("     their startup); mlsys26 charges head+slope*repeat+tail PER op, so it overcounts")
    print("     fused vector chains by ~(NOPS-1)*(head+tail). Per-op slope also diverges (div=4,")
    print("     cheap ops=1) vs mlsys26's single slope=2. *** grounding gap for the vector model.")


def analyze_reduce():
    print("\n=== reduce: TROWSUM/TCOLSUM are barrier-separated trees, not a single slope*repeat ===")
    idx = _index("reduce")

    # (A) TROWSUM: cost vs the reduced dim COLS (fixed ROWS) -- linear in COLS/64 (tree depth)
    r0 = idx["rs_cols"][0]["rows"]
    print(f"  (A) TROWSUM COLS sweep (ROWS={r0}): vec_cycles ~ 45*(COLS/64) + 6 (the vadd tree)")
    print(f"      {'COLS':>5} {'K':>3} | {'sim_vec':>7} {'pred':>6} {'e%':>5} | {'mlsys26':>7} {'over':>5}")
    for x in sorted(idx["rs_cols"], key=lambda r: r["cols"]):
        sim = C.read_aiv(x["fid"])["vec"]
        k = x["cols"] // (C.VEC_REG_BYTES // 4)
        pred = C.perfsim_trowsum_cycles(x["cols"])
        mly = C.mlsys_reduce_cycles(x["rows"], x["cols"])
        err = (sim - pred) / pred * 100 if pred else float("nan")
        print(f"      {x['cols']:>5} {k:>3} | {sim:>7} {pred:>6.0f} {err:>+4.1f}% | "
              f"{mly:>7.0f} {mly / sim:>4.1f}x")

    # (B) TROWSUM: cost vs ROWS (fixed COLS) -- the headline: count-mode -> ROWS-INDEPENDENT
    c0 = idx["rs_rows"][0]["cols"]
    print(f"  (B) TROWSUM ROWS sweep (COLS={c0}): sim is FLAT (count-mode, ROWS-independent);")
    print(f"      mlsys26's repeat = ROWS*COLS/64 grows with ROWS -> blows up on tall tiles.")
    print(f"      {'ROWS':>5} | {'sim_vec':>7} {'pred':>6} | {'mlsys26':>7} {'over':>5}")
    for x in sorted(idx["rs_rows"], key=lambda r: r["rows"]):
        sim = C.read_aiv(x["fid"])["vec"]
        pred = C.perfsim_trowsum_cycles(x["cols"])
        mly = C.mlsys_reduce_cycles(x["rows"], x["cols"])
        print(f"      {x['rows']:>5} | {sim:>7} {pred:>6.0f} | {mly:>7.0f} {mly / sim:>4.1f}x")

    # (C) TCOLSUM binary: pairwise vadd tree across rows (reduce H) -- scales with ROWS
    print("  (C) TCOLSUM binary ROWS sweep (COLS=128): vadd tree across rows ~ 16(R-1)+30*log2(R)")
    print(f"      {'ROWS':>5} | {'sim_vec':>7} {'pred':>6} {'e%':>5} | {'mlsys26':>7} {'over':>5}")
    for x in sorted(idx["cs_rows"], key=lambda r: r["rows"]):
        sim = C.read_aiv(x["fid"])["vec"]
        pred = C.perfsim_tcolsum_cycles(x["rows"])
        mly = C.mlsys_reduce_cycles(x["rows"], x["cols"])
        err = (sim - pred) / pred * 100 if pred else float("nan")
        print(f"      {x['rows']:>5} | {sim:>7} {pred:>6.0f} {err:>+4.1f}% | {mly:>7.0f} {mly / sim:>4.1f}x")

    print("  -> a reduction is a TREE of barrier-isolated count-mode passes (each re-pays head+")
    print("     tail), NOT one slope_reduce*repeat op. TROWSUM is ROWS-INDEPENDENT (count mode")
    print("     zeroes the repeat) and linear in COLS/64; mlsys26's repeat=ROWS*COLS/64 is")
    print("     structurally wrong (wrong ROWS scaling + magnitude). *** grounding gap. (NOTE: the")
    print("     perf-sim's count-mode flat-per-pass is itself coarse vs real HW -- flag for device.)")


def analyze_stream():
    print("\n=== stream: UB-overflow softmax is ONLINE (~1 wide pass), not #reductions+1 re-reads ===")
    idx = _index("stream")
    rows = sorted(idx["sweep"], key=lambda r: r["nchunks"])
    mat = C.read_aiv(rows[0]["fid"])["vec"]   # NCHUNKS=1 = the materialized single-pass baseline
    n_passes_mlsys = 3   # softmax has 2 reductions (rowmax, rowsum) -> mlsys26 N_passes=#red+1=3
    print(f"  softmax [{idx['rows']},{idx['cols']}] streamed over COLS; materialized (N=1) vec={mat}")
    print(f"  {'NCHUNKS':>7} {'CW':>5} | {'sim_vec':>7} {'ratio':>6} | {'mlsys26 3x':>10} {'over':>5}")
    for x in rows:
        sim = C.read_aiv(x["fid"])["vec"]
        mlsys = n_passes_mlsys * mat
        print(f"  {x['nchunks']:>7} {x['cols'] // x['nchunks']:>5} | {sim:>7} {sim / mat:>5.2f}x | "
              f"{mlsys:>10} {mlsys / sim:>4.1f}x")
    print("  -> online streaming runs the wide-body exp ONCE per element (the per-chunk reductions")
    print("     re-split the SAME total work + a thin [H,1] max/sum rescale), so the ratio stays")
    print("     well under mlsys26's flat 3x. mlsys26 multiplies compute by #reductions+1 on UB")
    print("     overflow -> ~3x pessimistic on every streamed softmax (large-context attention).")
    print("     *** grounding gap: its own comment asks for 'per-op liveness' -- wide-body factor")
    print("     ~1, surcharge = per-chunk re-paid startup + O(NCHUNKS) thin correction, not x3.")


ALL = {
    "vec_pointwise": analyze_pointwise,
    "vec_reduce": analyze_reduce,
    "vec_stream": analyze_stream,
}
