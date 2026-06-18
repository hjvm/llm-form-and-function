#!/usr/bin/env python3
"""One-command driver for the *Measuring Form and Function* pipeline.

Runs the five steps in dependency order; notebooks execute headless via nbconvert.
Run from the repository root.

    python run_pipeline.py                # full pipeline (Steps 1-5)
    python run_pipeline.py --from 2       # skip inference; start at the human baseline
    python run_pipeline.py --only 3,4     # run just those steps
    python run_pipeline.py --force        # pass --force to Step 1 (re-run inference)
    python run_pipeline.py --list         # print the plan and exit

Step 1 (inference, all 49 models) is the expensive one; once ``output/`` is populated,
re-run downstream analysis with ``--from 2``. ``build_annotation_sample.py`` is an optional
downstream extra and is not part of the default run.
"""
import argparse
import subprocess
import sys
import time

_NB = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace"]

# (name, command) in dependency order. See docs/REORG.md.
STEPS = [
    ("Model inference (CAC)",            [sys.executable, "run_experiments.py", "discourse",
                                          "--models", "model_configs.json"]),
    ("Human baseline + exports",         _NB + ["human_baseline.ipynb"]),
    ("Analytical metrics (overlap+TPR)", [sys.executable, "compute_analytical_metrics.py"]),
    ("Model TPR analysis",               _NB + ["lm_tpr_analysis.ipynb"]),
    ("Paper artifacts",                  _NB + ["analysis.ipynb"]),
]


def main():
    ap = argparse.ArgumentParser(description="Run the Form-and-Function pipeline (Steps 1-5).")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step (1-5)")
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS), help="last step (1-5)")
    ap.add_argument("--only", help="comma-separated step numbers, e.g. 3,4")
    ap.add_argument("--force", action="store_true", help="pass --force to Step 1 (re-run inference)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    selected = (sorted({int(x) for x in a.only.split(",")}) if a.only
                else list(range(a.start, a.end + 1)))
    selected = [n for n in selected if 1 <= n <= len(STEPS)]

    if a.list:
        for n in range(1, len(STEPS) + 1):
            print(f"  [{'x' if n in selected else ' '}] Step {n}: {STEPS[n - 1][0]}")
        return

    for n in selected:
        name, cmd = STEPS[n - 1]
        run = cmd + (["--force"] if (n == 1 and a.force) else [])
        print(f"\n{'=' * 70}\n[Step {n}/{len(STEPS)}] {name}\n  $ {' '.join(run)}\n{'=' * 70}")
        t = time.time()
        if subprocess.run(run).returncode != 0:
            sys.exit(f"\n!! Step {n} ({name}) failed. Stopping.")
        print(f"[Step {n}] done in {time.time() - t:.0f}s")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
