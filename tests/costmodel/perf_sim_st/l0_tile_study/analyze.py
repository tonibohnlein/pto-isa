# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Analysis for the L0-tile cost-model study: reads the perf-sim CSVs and checks each
# experiment's prediction. Run via run.py (which builds + runs first). See README.md.

import json

import common as C


def _index(name):
    return json.loads((C.RESULTS_DIR / f"{name}_index.json").read_text())


# ----------------------------------------------------------------------------- sweep
def analyze_sweep():
    print("\n=== sweep: CUBE closed-form + L0A/L0B asymmetry (split-K) ===")
    rows = _index("sweep")
    cube_ok = True
    mte1_of = {}
    by_label = {}
    for x in rows:
        d = C.read_aic(x["id"])
        pred = (x["M"] // x["m"]) * (x["N"] // x["n"]) * (x["K"] // x["k"]) * C.mad_cycles(x["m"], x["k"], x["n"])
        cube_ok &= d["cube"] == pred
        mte1_of[(x["label"], x["m"], x["k"], x["n"])] = d["mte1"]
        by_label.setdefault(x["label"], []).append((max(d["mte1"], d["cube"], d["fixp"]), x))
    print(f"  CUBE closed-form exact on all {len(rows)} tiles: {cube_ok}")
    violations = 0
    seen = set()
    for (label, m, k, n), v in sorted(mte1_of.items()):
        if m == n:
            continue
        key = (label, k, frozenset((m, n)))
        t = mte1_of.get((label, n, k, m))
        if t is None or key in seen:
            continue
        seen.add(key)
        tall, wide = (v, t) if m > n else (t, v)
        violations += tall > wide
    print(f"  tall (m>n) tile never loses to its transpose on MTE1: {violations == 0} ({violations} violations)")


# ----------------------------------------------------------------------------- fullk
def analyze_fullk():
    print("\n=== fullk: reuse saving + bandwidth-weighted stationary choice ===")
    by_tag = {}
    for x in _index("fullk"):
        by_tag.setdefault(x["tag"], {})[x["role"]] = x
    reuse_ok = agree = diff_bytes = total = 0
    print(f"  {'tag':>20} | {'base':>6} {'astat':>6} {'bstat':>6} | {'sim':>5} {'reuse%':>7} {'bytes':>6} {'time':>6}")
    for tag, r in sorted(by_tag.items()):
        x = r["base"]
        M, N, K, bm, bn = x["M"], x["N"], x["K"], x["baseM"], x["baseN"]
        mL, nL = M // bm, N // bn
        bm_, as_, bs_ = (C.read_aic(r[k]["id"])["mte1"] for k in ("base", "astat", "bstat"))
        cA, cB = bm * K * 2, K * bn * 2
        t_as = mL * cA / C.BW_L1_L0A + mL * nL * cB / C.BW_L1_L0B
        t_bs = nL * cB / C.BW_L1_L0B + mL * nL * cA / C.BW_L1_L0A
        b_as, b_bs = mL * cA + mL * nL * cB, nL * cB + mL * nL * cA
        sim = "astat" if as_ < bs_ else "bstat"
        time = "astat" if t_as < t_bs else "bstat"
        byts = "astat" if b_as < b_bs else ("bstat" if b_bs < b_as else "TIE")
        total += 1
        reuse_ok += as_ < bm_ and bs_ < bm_
        agree += sim == time
        diff_bytes += byts != time
        print(f"  {tag:>20} | {bm_:>6} {as_:>6} {bs_:>6} | {sim:>5} "
              f"{(bm_ - min(as_, bs_)) / bm_ * 100:>6.1f}% {byts:>6} {time:>6}")
    print(f"  reuse cuts MTE1 (both variants): {reuse_ok}/{total};  "
          f"sim==bandwidth-weighted: {agree}/{total};  bytes-only mis-picks: {diff_bytes}/{total}")


# ------------------------------------------------------------------------------- dbc
def analyze_dbc():
    print("\n=== dbc: L0C double-buffering (wall-clock) ===")
    by_tag = {}
    for x in _index("dbc"):
        by_tag.setdefault(x["tag"], {})[x["role"]] = x
    for tag, r in sorted(by_tag.items()):
        d = {role: C.read_aic(r[role]["id"]) for role in r}
        s1, s2, b1 = d["small1"]["total"], d["small2"]["total"], d["big1"]["total"]
        print(f"  {tag:>14}: fixed-tile single {s1} -> double {s2} ({(s1 - s2) / s1 * 100:+.1f}%);  "
              f"best-DB {s2} vs best-single {b1} -> {'DB' if s2 < b1 else 'single'} wins "
              f"{abs(s2 - b1) / max(s2, b1) * 100:.1f}%")


# -------------------------------------------------------------------------- accblock
def analyze_accblock():
    print("\n=== accblock (variant 3): A-reuse via NACC L0C accumulators ===")
    by_tag = {}
    for x in _index("accblock"):
        by_tag.setdefault(x["tag"], []).append(x)
    for tag, xs in sorted(by_tag.items()):
        xs.sort(key=lambda x: x["nacc"])
        base = None
        cells = []
        for x in xs:
            d = C.read_aic(x["id"])
            if base is None:
                base = d["mte1"]
            cells.append(f"NACC={x['nacc']}: mte1={d['mte1']} ({d['mte1'] / base * 100:.0f}%)")
        print(f"  {tag:>20}: " + "  ".join(cells))


# --------------------------------------------------------------------------- asymbuf
def analyze_asymbuf():
    print("\n=== asymbuf (variant 4): double-buffer the moving operand (full-K) ===")
    by_tag = {}
    for x in _index("asymbuf"):
        by_tag.setdefault(x["tag"], {})[x["movdb"]] = x
    for tag, r in sorted(by_tag.items()):
        d1, d2 = C.read_aic(r[1]["id"]), C.read_aic(r[2]["id"])
        same_mte1 = d1["mte1"] == d2["mte1"]
        print(f"  {tag:>20}: single-move total {d1['total']} -> double-move {d2['total']} "
              f"({(d1['total'] - d2['total']) / d1['total'] * 100:+.1f}%);  MTE1 unchanged: {same_mte1}")


ALL = {
    "gemm_sweep": analyze_sweep,
    "gemm_fullk": analyze_fullk,
    "gemm_dbc": analyze_dbc,
    "gemm_accblock": analyze_accblock,
    "gemm_asymbuf": analyze_asymbuf,
}
