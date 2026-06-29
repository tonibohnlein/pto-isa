# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Analysis for the GM->L1-tile cost-model study: reads the perf-sim CSVs and checks each
# experiment's prediction against our mlsys26 cube model. Run via run.py. See README.md.

import json

import common as C


def _index(name):
    return json.loads((C.RESULTS_DIR / f"{name}_index.json").read_text())


def _eff_bw_gibs(byts, cycles):
    """Back out the effective GiB/s the sim charged: bytes/2**30 * freq / cycles."""
    if cycles <= 0:
        return float("nan")
    return byts / (1024.0 ** 3) * C.FREQ_HZ / cycles


# --------------------------------------------------------------------------- reload
def analyze_reload():
    print("\n=== reload: MTE2 (GM->L1) == model reload bytes / BW_GM_L1 ? ===")
    rows = _index("reload")
    print(f"  {'tile (bmxbn)':>12} | {'reload_MiB':>10} {'sim_mte2':>9} {'pred@135':>9} "
          f"{'err%':>6} {'eff_GiB/s':>9} | bound")
    bws, errs = [], []
    for x in sorted(rows, key=lambda r: (r["bm"], r["bn"])):
        d = C.read_aic(x["id"])
        byts = C.reload_bytes(x["M"], x["N"], x["K"], x["bm"], x["bn"])
        pred = C.transfer_cycles(byts, C.BW_GM_L1)
        eff = _eff_bw_gibs(byts, d["mte2"])
        err = (d["mte2"] - pred) / pred * 100 if pred else float("nan")
        bws.append(eff)
        errs.append(abs(err))
        bound = max(("MTE2", d["mte2"]), ("MTE1", d["mte1"]), ("CUBE", d["cube"]),
                    ("FIXP", d["fixp"]), key=lambda t: t[1])[0]
        tile = f"{x['bm']}x{x['bn']}"
        print(f"  {tile:>12} | {byts / 2**20:>10.2f} "
              f"{d['mte2']:>9} {pred:>9.0f} {err:>+5.1f}% {eff:>9.1f} | {bound}")
    if bws:
        mean = sum(bws) / len(bws)
        spread = (max(bws) - min(bws)) / mean * 100
        print(f"  effective GM->L1 bandwidth: mean {mean:.1f} GiB/s, spread {spread:.1f}% "
              f"(flat table = {C.BW_GM_L1});  mean |err| vs flat: {sum(errs) / len(errs):.1f}%")
        print("  -> linear-in-bytes reload confirmed if spread is small; eff_bw reveals the"
              " sim's GM->L1 charge.")


# -------------------------------------------------------------------------- roofline
def analyze_roofline():
    print("\n=== roofline: total == max(mte2, mte1, cube, fixp) ? (overlap, not sum) ===")
    rows = _index("roofline")
    print(f"  {'regime':>18} | {'mte2':>8} {'mte1':>7} {'cube':>8} {'fixp':>7} | "
          f"{'max':>8} {'sum':>8} {'sim_total':>9} | bound  t/max  t/sum")
    for x in rows:
        d = C.read_aic(x["id"])
        pipes = [("MTE2", d["mte2"]), ("MTE1", d["mte1"]), ("CUBE", d["cube"]), ("FIXP", d["fixp"])]
        mx = max(pipes, key=lambda t: t[1])
        s = sum(p for _, p in pipes)
        t = d["total"]
        print(f"  {x['label']:>18} | {d['mte2']:>8} {d['mte1']:>7} {d['cube']:>8} {d['fixp']:>7} | "
              f"{mx[1]:>8} {s:>8} {t:>9} | {mx[0]:>4}  {t / mx[1]:>4.2f}  {t / s:>4.2f}")
    print("  -> t/max ~ 1 confirms the max-roofline (pipes overlap); t/sum << 1 confirms the\n"
          "     GM->L1 feed and the L0C->GM FixPipe drain are SEPARATE concurrent pipes.")


# ----------------------------------------------------------------------------- stepk
def analyze_stepk():
    print("\n=== stepk: MTE2 byte volume invariant to K-staging depth ? ===")
    rows = sorted(_index("stepk"), key=lambda r: r["step"])
    base = None
    print(f"  {'stepKa=stepKb':>14} | {'sim_mte2':>9} {'mte2 vs step1':>13} | {'sim_total':>9}")
    for x in rows:
        d = C.read_aic(x["id"])
        if base is None:
            base = d["mte2"]
        rel = d["mte2"] / base * 100 if base else float("nan")
        print(f"  {x['step']:>14} | {d['mte2']:>9} {rel:>12.1f}% | {d['total']:>9}")
    print("  -> mte2 flat across step confirms reload bytes are stepK-independent (our model"
          " omits stepK correctly); total may shift via TLOAD batching / overlap.")


ALL = {
    "gml1_reload": analyze_reload,
    "gml1_roofline": analyze_roofline,
    "gml1_stepk": analyze_stepk,
}
