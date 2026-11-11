#!/usr/bin/env python3
"""Run the *Measuring Form and Function* pipeline, or any part of it, in order.

    python run_pipeline.py --list        # show the steps and what each needs
    python run_pipeline.py --only 5      # paper tables + figures from committed results (no data needed)
    python run_pipeline.py --from 2      # everything after inference (needs data/ + model predictions)
    python run_pipeline.py               # everything, including ~2 weeks of model inference

Notebooks are executed headless with nbconvert and saved in place with their outputs.
Step 1 skips models whose predictions already exist; pass --force to re-run them.
Step 3 always recomputes, so its committed results are regenerated rather than reused.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
_NB = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.timeout=-1"]

# (name, needs, command)
STEPS = [
    ("Model inference (CAC)", "data/, GPU or Apple silicon, ~2 weeks",
     [sys.executable, str(PIPELINE / "01_run_inference.py")]),
    ("Human baselines + TPR cases", "data/",
     _NB + [str(PIPELINE / "02_human_baseline.ipynb")]),
    ("Analytical metrics (overlap + TPR)", "model predictions from Step 1, Step 2",
     [sys.executable, str(PIPELINE / "03_compute_metrics.py"), "--force"]),
    ("Model TPR tests", "model predictions from Step 1, Steps 2-3",
     _NB + [str(PIPELINE / "04_model_tpr.ipynb")]),
    ("Paper tables and figures", "committed results/ only",
     _NB + [str(PIPELINE / "05_paper_artifacts.ipynb")]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step (1-5)")
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS), help="last step (1-5)")
    ap.add_argument("--only", help="comma-separated step numbers, e.g. 3,4")
    ap.add_argument("--force", action="store_true", help="Step 1: re-run models whose predictions exist")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    a = ap.parse_args()

    selected = (sorted({int(x) for x in a.only.split(",")}) if a.only
                else list(range(a.start, a.end + 1)))
    selected = [n for n in selected if 1 <= n <= len(STEPS)]

    if a.list:
        for n, (name, needs, _) in enumerate(STEPS, start=1):
            print(f"  [{'x' if n in selected else ' '}] Step {n}: {name:<36} needs: {needs}")
        return

    for n in selected:
        name, _, cmd = STEPS[n - 1]
        run = cmd + (["--force"] if (n == 1 and a.force) else [])
        print(f"\n{'=' * 70}\n[Step {n}/{len(STEPS)}] {name}\n  $ {' '.join(run)}\n{'=' * 70}")
        t = time.time()
        if subprocess.run(run, cwd=ROOT).returncode != 0:
            sys.exit(f"\n!! Step {n} ({name}) failed. Stopping.")
        print(f"[Step {n}] done in {time.time() - t:.0f}s")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
