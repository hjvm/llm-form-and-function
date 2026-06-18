"""CAC — Contextual Alternative Choice.

Corpus/candidate/language-agnostic prompting and scoring for closed grammatical
alternations in language models. See ``cac.scoring.score_candidates``.
"""
from cac.context import (
    compute_context_starts,
    estimate_max_context_lines,
    format_context_input,
    reconstruct_context,
)
from cac.models import load_model
from cac.readout import choose_prediction, mass_on, surprisal
from cac.scoring import (
    AR_BLANK_TOKEN,
    get_default_device,
    merge_and_normalize,
    score_candidates,
)

__all__ = [
    "load_model",
    "score_candidates",
    "merge_and_normalize",
    "mass_on",
    "surprisal",
    "choose_prediction",
    "get_default_device",
    "AR_BLANK_TOKEN",
    "estimate_max_context_lines",
    "compute_context_starts",
    "reconstruct_context",
    "format_context_input",
]
