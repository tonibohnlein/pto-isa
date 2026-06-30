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


ALL = {
    "vec_pointwise": analyze_pointwise,
}
