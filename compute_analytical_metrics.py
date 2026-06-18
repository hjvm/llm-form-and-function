"""
Compute analytical D×N overlap and TPR metrics for all models.

Reads probability distributions stored by run_experiments.py (discourse
experiment) and computes all metrics analytically — no model inference needed.

Outputs
-------
Per-model overlap (one file per model):
    results/overlap/{type}/{model}/analytical_overlap_summary.csv
    Columns: model_name, model_type, child_name, speaker,
             emp_bias, empirical, naive_predicted, accuracy, N, S

Aggregate TPR (one file):
    results/tpr/analytical_tpr_all_results.csv
    Columns: model_name, model_type, child_name, speaker,
             tpr_overall, tpr_the, tpr_a, n_total, n_previous_a, n_previous_the

Usage
-----
    python compute_analytical_metrics.py
    python compute_analytical_metrics.py --discourse-dir output/manchester_discourse_childes
    python compute_analytical_metrics.py --force
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from determiner.analytical_metrics import (
    analytical_accuracy,
    analytical_bias,
    analytical_empirical_overlap,
    analytical_predicted_overlap,
    analytical_tpr,
    analytical_tpr_a,
    analytical_tpr_the,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DISCOURSE_DIR = "./output/manchester_discourse_childes"
DEFAULT_TPR_DIR       = "./output/manchester_tpr_childes"
DEFAULT_RESULTS_DIR   = "./results"
MODEL_TYPE_DIRS       = ["ar", "mlm", "seq2seq"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_predictions(model_dir: Path) -> pd.DataFrame:
    """Load all child and mother prediction CSVs for one model directory."""
    frames = []
    for child_dir in sorted(model_dir.iterdir()):
        if not child_dir.is_dir():
            continue
        for speaker in ["child", "mother"]:
            pred_file = child_dir / f"{speaker}_predictions.csv"
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                # Ensure speaker column reflects the file we loaded (some
                # older files may store 'CHI'/'MOT' instead of 'child'/'mother').
                df["speaker"] = speaker
                df["child_name"] = child_dir.name
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_overlap_rows(predictions: pd.DataFrame) -> list:
    """Compute analytical overlap metrics per (child_name, speaker)."""
    rows = []
    for (child_name, speaker), group in predictions.groupby(
        ["child_name", "speaker"], sort=False
    ):
        if group.empty or "noun" not in group.columns:
            continue
        sites = group.dropna(subset=["P_the", "P_a", "noun"])
        if sites.empty:
            continue

        row = {
            "model_name":      group["model_name"].iloc[0],
            "model_type":      group["model_type"].iloc[0],
            "child_name":      child_name,
            "speaker":         speaker,
            "N":               sites["noun"].nunique(),
            "S":               len(sites),
            "emp_bias":        analytical_bias(sites),
            "empirical":       analytical_empirical_overlap(sites),
            "naive_predicted": analytical_predicted_overlap(sites),
            "accuracy":        analytical_accuracy(sites),
        }
        rows.append(row)
    return rows


def _build_tpr_df(predictions: pd.DataFrame, tpr_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Join discourse predictions with TPR metadata to get eligible pairs.

    Join key: child_name + speaker + sentence_id (unique within child/speaker).
    Returns a DataFrame with one row per eligible transition pair, with columns:
        d_A      (int)   : 1 if prior human said 'the', 0 if 'a'
        P_the_B  (float) : model's P(the) at the target site
    """
    if predictions.empty or tpr_metadata.empty:
        return pd.DataFrame()

    # Normalize previous_det to binary d_A
    tpr_meta = tpr_metadata.copy()
    det_map = {"the": 1, "a": 0, "an": 0}
    tpr_meta["d_A"] = tpr_meta["previous_det"].str.lower().map(det_map)
    tpr_meta = tpr_meta.dropna(subset=["d_A"])
    tpr_meta["d_A"] = tpr_meta["d_A"].astype(int)

    # Normalize P_the
    pred = predictions.copy()
    total = pred["P_the"] + pred["P_a"]
    pred["P_the_B"] = pred["P_the"] / total.replace(0, np.nan)

    merged = tpr_meta.merge(
        pred[["child_name", "speaker", "sentence_id", "P_the_B"]],
        on=["child_name", "speaker", "sentence_id"],
        how="inner",
    )
    return merged


def _compute_tpr_rows(predictions: pd.DataFrame, tpr_metadata: pd.DataFrame) -> list:
    """Compute analytical TPR metrics per (child_name, speaker)."""
    paired = _build_tpr_df(predictions, tpr_metadata)
    if paired.empty:
        return []

    rows = []
    model_name = predictions["model_name"].iloc[0]
    model_type = predictions["model_type"].iloc[0]

    for (child_name, speaker), group in paired.groupby(
        ["child_name", "speaker"], sort=False
    ):
        if group.empty:
            continue
        prev_a   = group[group["d_A"] == 0]
        prev_the = group[group["d_A"] == 1]
        rows.append({
            "model_name":    model_name,
            "model_type":    model_type,
            "child_name":    child_name,
            "speaker":       speaker,
            "tpr_overall":   analytical_tpr(group),
            "tpr_the":       analytical_tpr_the(group),
            "tpr_a":         analytical_tpr_a(group),
            "n_total":       len(group),
            "n_previous_a":  len(prev_a),
            "n_previous_the":len(prev_the),
        })
    return rows


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(discourse_dir: str, tpr_dir: str, results_dir: str = DEFAULT_RESULTS_DIR,
        force: bool = False, skip_tpr: bool = False) -> None:
    discourse_path      = Path(discourse_dir)
    tpr_path            = Path(tpr_dir)
    results_overlap_dir = Path(results_dir) / "overlap"
    results_tpr_dir     = Path(results_dir) / "tpr"
    results_tpr_dir.mkdir(parents=True, exist_ok=True)

    tpr_metadata = pd.DataFrame()
    if not skip_tpr:
        tpr_metadata_file = tpr_path / "tpr_cached_inputs.csv"
        if not tpr_metadata_file.exists():
            print(f"ERROR: TPR metadata not found at {tpr_metadata_file}")
            print("Run:  python run_experiments.py tpr --models model_configs.json "
                  "--discourse-label-style childes")
            print("Or pass --skip-tpr to compute only overlap metrics.")
            sys.exit(1)
        print(f"Loading TPR metadata from {tpr_metadata_file} …")
        tpr_metadata = pd.read_csv(tpr_metadata_file)
        print(f"  {len(tpr_metadata)} eligible transition pairs\n")

    all_tpr_rows = []

    for model_type in MODEL_TYPE_DIRS:
        type_path = discourse_path / model_type
        if not type_path.exists():
            continue

        for model_dir in sorted(type_path.iterdir()):
            if not model_dir.is_dir():
                continue

            overlap_out = results_overlap_dir / model_type / model_dir.name / "analytical_overlap_summary.csv"
            overlap_out.parent.mkdir(parents=True, exist_ok=True)
            if overlap_out.exists() and not force:
                print(f"  [skip] {model_dir.name} (overlap exists, use --force to recompute)")
                predictions = _load_predictions(model_dir) if not skip_tpr else pd.DataFrame()
            else:
                predictions = _load_predictions(model_dir)
                if predictions.empty:
                    print(f"  [warn] {model_dir.name}: no prediction files found")
                    continue

                # --- Overlap ---
                overlap_rows = _compute_overlap_rows(predictions)
                if overlap_rows:
                    pd.DataFrame(overlap_rows).to_csv(overlap_out, index=False)
                    print(f"  [done] {model_dir.name}: overlap → {overlap_out.name}")
                else:
                    print(f"  [warn] {model_dir.name}: no overlap rows produced")

            # --- TPR ---
            if not skip_tpr and not predictions.empty:
                tpr_rows = _compute_tpr_rows(predictions, tpr_metadata)
                all_tpr_rows.extend(tpr_rows)

    if skip_tpr:
        print("\n[info] Skipped TPR computation (--skip-tpr).")
        return

    # Write aggregate TPR file
    tpr_out = results_tpr_dir / "analytical_tpr_all_results.csv"
    if all_tpr_rows:
        pd.DataFrame(all_tpr_rows).to_csv(tpr_out, index=False)
        n_models = pd.DataFrame(all_tpr_rows)["model_name"].nunique()
        print(f"\n[done] Analytical TPR → {tpr_out} ({n_models} models)")
    else:
        print("\n[warn] No TPR rows produced — check that discourse predictions "
              "and TPR metadata share sentence_id values.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--discourse-dir", default=DEFAULT_DISCOURSE_DIR,
                        help=f"Discourse output directory (default: {DEFAULT_DISCOURSE_DIR})")
    parser.add_argument("--tpr-dir", default=DEFAULT_TPR_DIR,
                        help=f"TPR output directory (default: {DEFAULT_TPR_DIR})")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Results directory for committed outputs (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if analytical_overlap_summary.csv already exists")
    parser.add_argument("--skip-tpr", action="store_true",
                        help="Skip TPR computation (useful when tpr_cached_inputs.csv is absent, e.g. isolated experiment)")
    args = parser.parse_args()
    run(args.discourse_dir, args.tpr_dir, results_dir=args.results_dir,
        force=args.force, skip_tpr=args.skip_tpr)


if __name__ == "__main__":
    main()
