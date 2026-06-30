# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Analysis for the MIXED cube+vector tile cost-model study: reads the perf-sim CSVs and grounds
# the overlap/serial behaviour of a tiled matmul->pointwise pipeline. Run via run.py.
#
# Cycle accounting (reporter_core_impl.inl) -- the AIC row carries only cube pipes (mte2_aic/mte1/
# cube/fixp), the AIV0 row only vector pipes (mte2_aiv/vec/mte3):
#   busy = sum of that unit's pipes -- OVER-counts the wall (the pipes pipeline internally).
#   span = active_end - active_start.
# The per-UNIT continuous wall is the active SPAN measured in the OVERLAP run, where each unit
# runs back-to-back with no cross-unit idle (cube=AICspan, vec=AIVspan). The first-tile FILL is
# the AIV active_start in the overlap run (the vector waits one cube tile before it can begin).
# Both tables below are computed by PAIRING each (shape, NTILES) overlap run with its serial twin.

import json

import common as C


def _index(name):
    return json.loads((C.RESULTS_DIR / f"{name}_index.json").read_text())


def _pairs():
    """Merge the overlap + serial sweeps by (bm, K, N, NTILES). Returns rows with both measured
    totals and the per-unit continuous walls/fill taken from the OVERLAP run."""
    ov = {(r["bm"], r["K"], r["N"], r["ntiles"]): r for r in _index("mixed_overlap")["sweep"]}
    se = {(r["bm"], r["K"], r["N"], r["ntiles"]): r for r in _index("mixed_serial")["sweep"]}
    out = []
    for key in sorted(ov, key=lambda k: (k[0], k[1], k[2], k[3])):
        o = C.read_mixed(ov[key]["fid"])
        s = C.read_mixed(se[key]["fid"]) if key in se else None
        bm, K, N, nt = key
        cube_wall = o["aic"]["active"]     # continuous cube wall (overlap run span)
        vec_wall = o["aiv"]["active"]      # continuous vector wall
        fill = o["aiv"]["start"]           # one cube tile -- vector cannot start until tile 0 lands
        out.append(dict(bm=bm, K=K, N=N, nt=nt, cube_wall=cube_wall, vec_wall=vec_wall, fill=fill,
                        ov_total=o["total"], se_total=(s["total"] if s else None),
                        ov_busy=(o["aic"]["busy"], o["aiv"]["busy"]),
                        ddr=C.ddr_cycles(nt * bm, N, K, bm, N)))
    return out


def _shapes(rows):
    seen = []
    for r in rows:
        k = (r["bm"], r["K"], r["N"])
        if k not in seen:
            seen.append(k)
    return seen


def _print_table(rows, schedule):
    """schedule in {'overlap','serial'}: pick which measured total to ground."""
    for bm, K, N in _shapes(rows):
        sweep = [r for r in rows if (r["bm"], r["K"], r["N"]) == (bm, K, N)]
        print(f"\n=== mixed_{schedule}: tile [bm={bm}, N={N}, K={K}]  (fp16 in / fp32 acc, C=C+C) ===")
        print(f"  {'NT':>3} | {'cubeW':>6} {'vecW':>6} {'fill':>5} | {'total':>6} {'serial':>6} "
              f"{'pipe':>6} {'mlsys':>6} | {'t/srl':>5} {'t/pipe':>6} {'ovl':>4}")
        for r in sweep:
            cw, vw, fill, nt = r["cube_wall"], r["vec_wall"], r["fill"], r["nt"]
            total = r["ov_total"] if schedule == "overlap" else r["se_total"]
            if total is None:
                continue
            serial = C.predict_serial(cw, vw)           # cube + vec (pipelined-rate sum)
            pipe = C.predict_pipelined(cw, vw, fill)    # max(cube,vec) + one tile fill
            mlsys = C.mlsys_mixed_latency(cw, vw, r["ddr"])  # max(cube, vec, ddr) -- no fill term
            ovl = C.overlap_factor(cw, vw, total)
            print(f"  {nt:>3} | {cw:>6} {vw:>6} {fill:>5} | {total:>6} {serial:>6.0f} "
                  f"{pipe:>6.0f} {mlsys:>6.0f} | {total / serial:>5.2f} {total / pipe:>6.2f} {ovl:>4.2f}")


def analyze_overlap():
    print("\n########## mixed_overlap: SKEWED ping-pong producer (cube one tile ahead) ##########")
    rows = _pairs()
    _print_table(rows, "overlap")
    print("\n  -> overlap total tracks pipe = max(cube,vec)+fill (t/pipe ~ 1), well below the serial")
    print("     sum (t/srl < 1, shrinking with NT). mlsys = max(cube,vec,ddr) OMITS the one-tile")
    print("     fill, so it under-reads by `fill` (matters only at small NT). overlap_factor -> ~1.")
    _fill_scaling(rows, "overlap")


def analyze_serial():
    print("\n########## mixed_serial: single buffer + B-operand chain (cube waits prior vector) ##########")
    rows = _pairs()
    _print_table(rows, "serial")
    print("\n  -> serial total tracks the SUM (t/srl >= 1; it even exceeds cube+vec because isolating")
    print("     each tile also kills intra-AIC cross-tile pipelining), far above pipe. overlap_factor ~ 0.")
    _fill_scaling(rows, "serial")
    _combined(rows)


def _fill_scaling(rows, schedule):
    print(f"  fill/drain scaling ({schedule}): fill = one cube tile (constant in NT) -> its SHARE shrinks")
    for bm, K, N in _shapes(rows):
        parts = []
        for r in [x for x in rows if (x["bm"], x["K"], x["N"]) == (bm, K, N)]:
            total = r["ov_total"] if schedule == "overlap" else r["se_total"]
            if total:
                parts.append(f"NT{r['nt']}={r['fill'] / total * 100:.0f}%")
        print(f"    [bm={bm},N={N},K={K}] fill share of total: " + "  ".join(parts))


def _combined(rows):
    print("\n  === overlap vs serial on identical work (wall-clock ground truth) ===")
    for bm, K, N in _shapes(rows):
        print(f"  [bm={bm}, N={N}, K={K}]")
        print(f"    {'NT':>3} | {'overlap':>7} {'serial':>7} {'speedup':>7} | {'ovl(ov)':>7} {'ovl(se)':>7}")
        for r in [x for x in rows if (x["bm"], x["K"], x["N"]) == (bm, K, N)]:
            if r["se_total"] is None:
                continue
            sp = r["se_total"] / r["ov_total"] if r["ov_total"] else float("nan")
            ovl_o = C.overlap_factor(r["cube_wall"], r["vec_wall"], r["ov_total"])
            ovl_s = C.overlap_factor(r["cube_wall"], r["vec_wall"], r["se_total"])
            print(f"    {r['nt']:>3} | {r['ov_total']:>7} {r['se_total']:>7} {sp:>6.2f}x | "
                  f"{ovl_o:>7.2f} {ovl_s:>7.2f}")
    print("    NT=1: overlap==serial (a single tile cannot overlap -- 100% fill). speedup & overlap")
    print("    grow with NT toward (cube+vec)/max as the one-tile fill/drain amortizes.")


def _dom_aic(aic):
    """Dominant AIC pipe + its 'bound' class. cube=MAD (compute); mte2_aic/fixp/mte1=GM (data)."""
    pipes = {"cube": aic["cube"], "mte2_aic": aic["mte2"], "fixp": aic["fixp"], "mte1": aic["mte1"]}
    name = max(pipes, key=pipes.get)
    return name, pipes[name], ("cube" if name == "cube" else "gm")


def _dom_aiv(aiv):
    pipes = {"mte2_aiv": aiv["mte2"], "vec": aiv["vec"], "mte3": aiv["mte3"]}
    name = max(pipes, key=pipes.get)
    return name, pipes[name]


def analyze_ddr_bound():
    print("\n########## mixed_ddr_bound: K sweep (bm=128, NT=8) -- is `ddr` a separate max term? ##########")
    rows = _index("mixed_ddr_bound")["sweep"]
    Ns = []
    for r in rows:
        if r["N"] not in Ns:
            Ns.append(r["N"])
    for N in Ns:
        sweep = sorted((r for r in rows if r["N"] == N), key=lambda r: r["K"])
        bm, nt = sweep[0]["bm"], sweep[0]["ntiles"]
        print(f"\n=== mixed_ddr_bound: N={N}, bm={bm}, NT={nt}  (C=C+C; AIV stage = GM load+store) ===")
        print(f"  {'K':>4} | {'MAD':>5} {'cubeW':>6} {'AICdom':>16} | {'vecW':>5} {'AIVdom':>14} | "
              f"{'ddr':>5} | {'total':>6} {'max+fill':>8} {'t/(m+f)':>7} {'mlsys':>6} | {'stage':>5} {'aic':>4}")
        for r in sweep:
            m = C.read_mixed(r["fid"])
            cw, vw, fill = m["aic"]["active"], m["aiv"]["active"], m["aiv"]["start"]
            an, av, abound = _dom_aic(m["aic"])
            vn, vv = _dom_aiv(m["aiv"])
            ddr = C.ddr_cycles(nt * bm, N, r["K"], bm, N)
            total = m["total"]
            mf = C.predict_pipelined(cw, vw, fill)
            mlsys = C.mlsys_mixed_latency(cw, vw, ddr)
            stage = "cube" if cw >= vw else "vec"   # which UNIT is the bottleneck stage
            print(f"  {r['K']:>4} | {m['aic']['cube']:>5} {cw:>6} {an + '(' + str(av) + ')':>16} | {vw:>5} "
                  f"{vn + '(' + str(vv) + ')':>14} | {ddr:>5.0f} | {total:>6} {mf:>8.0f} "
                  f"{total / mf:>7.2f} {mlsys:>6.0f} | {stage:>5} {abound:>4}")
        print("  -> total tracks max(cube_stage,vec_stage)+fill (t/(m+f) ~ 1) across the WHOLE sweep, and")
        print("     mlsys=max(cube,vec,ddr) == max(cube,vec): the `ddr` (max GM port) is always <= the")
        print("     stage that subsumes it -- never a separate max term.")
        print("     Two crossovers, both ~K=128: (1) bottleneck STAGE flips vec->cube (MAD grows with K,")
        print("     the C=C+C vector stage is K-independent GM load+store); (2) the AIC's dominant pipe")
        print("     flips fixp(store, K-indep) -> mte2_aic(reload). NOTE the AIC stays GM-bound (aic=gm)")
        print("     throughout: MAD grows but never overtakes reload, because each tile RELOADS B[K,N]")
        print("     (reload ~ K tracks MAD ~ K). A B-resident kernel would push it to aic=cube at large K.")


def analyze_contention():
    print("\n########## mixed_contention: do cube (GM->L1) + vector (GM->UB) reads share ONE HBM pool? ##########")
    meta = _index("mixed_contention")
    hbm, bm, K, N, nt = meta["hbm"], meta["bm"], meta["K"], meta["N"], meta["nt"]
    pk_c, pk_v = C.BW_GM_L1, C.BW_GM_UB         # cube reload peak 135, vector load peak 100.9
    knee_c, knee_v = hbm / pk_c, hbm / pk_v
    print(f"  skewed mixed kernel [bm={bm}, N={N}, K={K}, NT={nt}], multi-core, read pool={hbm:.0f} GiB/s")
    print(f"  cube GM->L1 peak={pk_c:.0f} (knee {knee_c:.1f} cores); vector GM->UB peak={pk_v:.1f} (knee {knee_v:.1f})")
    print(f"  {'B':>3} | {'cube mte2 un->cap':>17} {'bw_c':>6} {'pred':>5} | {'vec mte2 un->cap':>16} {'bw_v':>6} {'pred':>5}"
          f" | {'900/B':>6} | {'wall_un':>7} {'wall_cap':>8} {'delta':>6} {'pool':>5}")
    errs, bind_c, bind_v = [], None, None
    for x in sorted(meta["sweep"], key=lambda r: r["B"]):
        B = x["B"]
        un, cap = C.read_mixed(x["un"]), C.read_mixed(x["cap"])
        cu, cc = un["aic"]["mte2"], cap["aic"]["mte2"]    # cube reload mte2_aic (un, cap)
        vu, vc = un["aiv"]["mte2"], cap["aiv"]["mte2"]    # vector load mte2_aiv (un, cap)
        bw_c = pk_c * cu / cc if cc else float("nan")     # back out effective BW (pure ratio)
        bw_v = pk_v * vu / vc if vc else float("nan")
        prd_c, prd_v = min(pk_c, hbm / B), min(pk_v, hbm / B)
        errs += [abs(bw_c - prd_c) / prd_c * 100, abs(bw_v - prd_v) / prd_v * 100]
        if bind_c is None and cc > cu * 1.02:
            bind_c = B
        if bind_v is None and vc > vu * 1.02:
            bind_v = B
        # "pool" = both read pipes throttled to the SAME 900/B (one shared pool) once past both knees
        pool = "yes" if (B > knee_v and abs(bw_c - bw_v) / bw_v < 0.05) else ("cube" if B > knee_c else "-")
        delta = cap["total"] - un["total"]
        print(f"  {B:>3} | {str(cu) + '->' + str(cc):>17} {bw_c:>6.1f} {prd_c:>5.0f} | "
              f"{str(vu) + '->' + str(vc):>16} {bw_v:>6.1f} {prd_v:>5.0f} | {hbm / B:>6.1f} | "
              f"{un['total']:>7} {cap['total']:>8} {delta:>6} {pool:>5}")
    print(f"  -> (Q1) FLAT/uncapped: per-core mte2 (cube {C.read_mixed(meta['sweep'][0]['un'])['aic']['mte2']}, "
          f"vector {C.read_mixed(meta['sweep'][0]['un'])['aiv']['mte2']}) is CONSTANT in B, wall_un flat -- no shared cap.")
    print(f"     (Q2) CAPPED: cube read BW drops to 900/B from B={bind_c} (knee {knee_c:.1f}), vector from "
          f"B={bind_v} (knee {knee_v:.1f}); back-out matches min(peak,900/B) to {max(errs):.0f}% max, wall grows.")
    print(f"     (Q3) CROSS-UNIT: ONE total_read_gibs knob (GroupTotal pools GM_TO_L1 + GM_TO_UB) throttles")
    print(f"     BOTH the cube AND vector reads of the SAME mixed kernel -- past the knees both collapse to")
    print(f"     900/B (pool=yes). The single shared read pool the mlsys26 ddr_lat/par() term must model;")
    print(f"     it is applied per-pipe-per-core as min(peak, 900/B), not as a summed-volume budget.")


ALL = {
    "mixed_overlap": analyze_overlap,
    "mixed_serial": analyze_serial,
    "mixed_ddr_bound": analyze_ddr_bound,
    "mixed_contention": analyze_contention,
}


if __name__ == "__main__":
    # Re-analyze existing CSVs (run.py --no-build calls these too).
    for _fn in ALL.values():
        _fn()
