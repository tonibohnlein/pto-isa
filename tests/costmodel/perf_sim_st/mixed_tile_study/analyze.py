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
                        ddr=C.ddr_cycles(nt * bm, N, K, bm, N, K)))
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


ALL = {
    "mixed_overlap": analyze_overlap,
    "mixed_serial": analyze_serial,
}


if __name__ == "__main__":
    # Re-analyze existing CSVs (run.py --no-build calls these too).
    analyze_overlap()
    analyze_serial()
