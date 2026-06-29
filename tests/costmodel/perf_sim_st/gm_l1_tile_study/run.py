#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# One-command reproduce for the GM->L1-tile cost-model study:
#   generate testcases -> build under __COSTMODEL -> run -> analyze.
#
# Usage:
#   python run.py                       # all experiments, full build + analyze
#   python run.py gml1_reload           # one experiment
#   python run.py --no-build            # re-analyze existing CSVs only
#   python run.py --build-dir /path/bd  # use a specific cmake build dir
#
# Requires: cmake, a C++23 compiler, GTest. No NPU -- the kernels build under
# -D__COSTMODEL and run the host pipeline simulator. The testcase names produced by
# experiments.py must be registered in ../testcase/CMakeLists.txt (ALL_TESTCASES).

import argparse
import os
import pathlib
import subprocess
import sys

import analyze
import common as C
import experiments

REPO_ROOT = C.PERF_SIM_ROOT.parents[2]
DEFAULT_BUILD = C.STUDY_DIR / "build"


def sh(cmd, cwd=None, env=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def ensure_formula_headers():
    """The cost model needs generated formula-param headers (CSV -> .hpp, no args)."""
    for arch in ("a2a3", "a5"):
        gen = REPO_ROOT / "include/pto/costmodel" / arch / "formula_costmodel/gen_formula_params_header.py"
        hdr = gen.parent / "formula_params_generated.hpp"
        if gen.exists() and not hdr.exists():
            sh([sys.executable, str(gen)], cwd=REPO_ROOT)


def configure(build_dir):
    if not (build_dir / "CMakeCache.txt").exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        sh(["cmake", "-S", str(C.PERF_SIM_ROOT), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release", "-DPTO_GLIBCXX_USE_CXX11_ABI=1"])
    else:
        sh(["cmake", str(build_dir)])  # reconfigure to pick up new testcases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=[], help="subset (default: all)")
    ap.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    ap.add_argument("--no-build", action="store_true", help="skip build/run; analyze existing CSVs")
    ap.add_argument("--fitted", action="store_true",
                    help="also run binaries under PTO_BW_MODE=fitted and analyze the Hill GM->L1 model")
    args = ap.parse_args()

    names = args.experiments or list(experiments.ALL)
    build_dir = pathlib.Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = C.STUDY_DIR / args.build_dir
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        print("== prepare ==")
        ensure_formula_headers()
        print("== generate testcases ==")
        for name in names:
            experiments.ALL[name]()
        print("== configure + build ==")
        configure(build_dir)
        for name in names:
            sh(["cmake", "--build", str(build_dir), "--target", name, "-j8"])
        print("== run (CSVs -> results/perf_sim_output) ==")
        for name in names:
            sh([str(build_dir / "bin" / name)], cwd=C.RESULTS_DIR)
        if args.fitted:
            print("== run fitted (PTO_BW_MODE=fitted, CSVs -> results/fitted/perf_sim_output) ==")
            fitted_cwd = C.RESULTS_DIR / "fitted"
            (fitted_cwd / "perf_sim_output").mkdir(parents=True, exist_ok=True)
            for name in names:
                sh([str(build_dir / "bin" / name)], cwd=fitted_cwd, env={**os.environ, "PTO_BW_MODE": "fitted"})

    print("== analyze ==")
    for name in names:
        analyze.ALL[name]()
    if args.fitted:
        analyze.analyze_fitted()


if __name__ == "__main__":
    main()
