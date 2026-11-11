#!/usr/bin/env python3
"""Step 1 — CAC inference: score `a` vs `the` at every Manchester D×N site, for every model.

Each model sees the full preceding session as discourse context (speaker labels
*CHI/*MOT), trimmed from the start to fit its context window. Writes one CSV per
model, dyad and speaker:

    output/manchester_discourse_childes/<type>/<model>/<child>/{child,mother}_predictions.csv

Models whose predictions already exist are skipped unless --force is given. A full run
over all 49 model evaluations takes about two weeks on an Apple-silicon laptop.

Usage:
    python pipeline/01_run_inference.py
    python pipeline/01_run_inference.py --models configs/models.json --force
"""
import argparse
import json

from determiner.corpus import DEFAULT_CORPUS_PATH
from determiner.inference import run_discourse_experiment
from determiner.paths import DISCOURSE_DIR, MODEL_CONFIGS, chdir_to_root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default=str(MODEL_CONFIGS),
                        help='model list (JSON; default: configs/models.json)')
    parser.add_argument('--corpus', default=DEFAULT_CORPUS_PATH,
                        help=f'Manchester corpus directory (default: {DEFAULT_CORPUS_PATH})')
    parser.add_argument('--output', default=str(DISCOURSE_DIR),
                        help='prediction output directory (default: output/manchester_discourse_childes)')
    parser.add_argument('--force', action='store_true', help='re-run models whose predictions already exist')
    args = parser.parse_args()

    chdir_to_root()
    with open(args.models) as f:
        model_configs = json.load(f)
    print(f"Loaded {len(model_configs)} model configurations from {args.models}")

    run_discourse_experiment(model_configs, args.corpus, args.output,
                             model_configs_path=args.models, skip_existing=not args.force)
    print("\nInference complete.")


if __name__ == '__main__':
    main()
