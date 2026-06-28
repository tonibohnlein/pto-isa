# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Cross-comparison: for problems/tiles shared by the fullk and accblock experiments,
# put split-K vs full-K vs accumulator-blocking side by side (resident operands).
# Reports the MTE1 (L1->L0 traffic) each tiling achieves and the bounding pipe, so we
# can read off which algorithm shines in which regime. Run: python compare.py
# (requires the fullk + accblock CSVs from a prior `run.py`).

import json

import common as C


def _by_key(name):
    return json.loads((C.RESULTS_DIR / f"{name}_index.json").read_text())


def main():
    # full-K: best of A-/B-stationary, plus the split-K baseline, keyed by (M,N,K,m,n).
    fk = {}
    for x in _by_key("fullk"):
        k = (x["M"], x["N"], x["K"], x["baseM"], x["baseN"])
        fk.setdefault(k, {})[x["role"]] = C.read_aic(x["id"])
    # accblock: NACC=1 is split-K; best (largest NACC) is the reuse result.
    ab = {}
    for x in _by_key("accblock"):
        k = (x["M"], x["N"], x["K"], x["baseM"], x["baseN"])
        ab.setdefault(k, {})[x["nacc"]] = C.read_aic(x["id"])

    shared = sorted(set(fk) & set(ab))
    print("\n=== cross-comparison: split-K vs full-K vs accumulator-blocking ===")
    print("  (MTE1 = L1->L0 traffic, lower is better; full-K uses k==K, accblock splits K)")
    print(f"  {'M x N x K  tile':>22} | {'splitK':>7} {'fullK':>7} {'accblk':>7} | "
          f"{'best':>7} {'winner':>8} | bound(best)")
    print("  " + "-" * 86)
    for key in shared:
        M, N, Kk, m, n = key
        split_mte1 = fk[key]["base"]["mte1"]
        full = min((fk[key]["astat"], fk[key]["bstat"]), key=lambda d: d["mte1"])
        nacc_best = max(ab[key])
        acc = ab[key][nacc_best]
        cands = {"split-K": (split_mte1, fk[key]["base"]),
                 "full-K": (full["mte1"], full),
                 f"accbl{nacc_best}": (acc["mte1"], acc)}
        win = min(cands, key=lambda kk: cands[kk][0])
        wd = cands[win][1]
        bound = max(("MTE1", wd["mte1"]), ("CUBE", wd["cube"]), ("FIXP", wd["fixp"]), key=lambda t: t[1])
        print(f"  {f'{M}x{N}x{Kk} {m}x{n}':>22} | {split_mte1:>7} {full['mte1']:>7} {acc['mte1']:>7} | "
              f"{cands[win][0]:>7} {win:>8} | {bound[0]} ({bound[1]})  cube={wd['cube']} fixp={wd['fixp']}")
    print("\n  Reading: full-K gives the deepest reuse when k==K fits L0; accumulator-blocking\n"
          "  gives most of it while still splitting K; split-K is the no-reuse floor. Whether the\n"
          "  MTE1 saving moves the WALL depends on the bound -- if CUBE or FIXP dominates, reuse is\n"
          "  free headroom, not speedup.")


if __name__ == "__main__":
    main()
