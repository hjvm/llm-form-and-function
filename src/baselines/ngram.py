"""5-gram baselines, run through the paper's CAC pipeline exactly like the autoregressive LMs.

The n-grams are wrapped as ``ar`` scorers in ``src/cac/ngram.py`` and run with the unchanged
Step 1 (``pipeline/01_run_inference.py --models configs/ngram_models.json``): same discourse
prompts (preceding session lines labelled ``*CHI:``/``*MOT:``, target line with the blank), same
1,024-token budget, same scoring of the whole prompt with each form, same a/an merge and
softmax (``cac.scoring.merge_and_normalize``). Step 3 then computes overlap, bias, predicted
overlap and TPR from those predictions unchanged.

* ``ngram/aochildes-kn5``: interpolated Kneser-Ney 5-gram trained on AO-CHILDES (Huebner &
  Willits 2021; 893,989 American English child-directed utterances). Manchester is not in it.
* ``ngram/wikipedia-kenlm5``: the public KenLM model BramVanroy/kenlm_wikipedia_en.

This module only collects Step 1's predictions into one compact file per model
(``results/baselines/ngram/<model>_cac_predictions.csv.gz``: sentence_id, speaker, P_a, P_the),
so the baseline tables and figures can be rebuilt without re-running inference.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from determiner.paths import OUTPUT_DIR

DISCOURSE_AR = OUTPUT_DIR / "manchester_discourse_childes" / "ar"
NGRAM_MODELS = {"aochildes-kn5": "5-gram, child-directed speech (AO-CHILDES)",
                "wikipedia-kenlm5": "5-gram, Wikipedia (KenLM)"}


def collect_predictions(model: str, discourse_ar: Path = DISCOURSE_AR) -> pd.DataFrame:
    """Concatenate Step 1's {child,mother}_predictions.csv for one model."""
    files = sorted(Path(discourse_ar, model).glob("*/*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"no Step 1 predictions under {Path(discourse_ar, model)}")
    cols = ["sentence_id", "child_name", "speaker", "P_a", "P_the", "truncated"]
    return pd.concat([pd.read_csv(f, usecols=cols) for f in files]).sort_values("sentence_id")


def p_the(preds: pd.DataFrame, sentence_ids) -> np.ndarray:
    """P(the) at every site, in the order of ``sentence_ids``."""
    p = preds.set_index("sentence_id").P_the.reindex(sentence_ids).values
    assert not np.isnan(p).any(), "predictions do not cover every site"
    return p
