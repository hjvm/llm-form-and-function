"""Architecture-dispatched Contextual Alternative Choice (CAC) scoring.

Corpus/candidate/language-agnostic core. Given a prompt that contains a blank/mask
placeholder and a **closed candidate set** expressed as categories -> surface forms,
return a normalized probability distribution over the *categories*.

``candidates`` is a mapping ``{category_label: [surface_form, ...]}``. Within a
category the member forms' log-scores are merged in log-space (``logaddexp`` =
summing probability mass); the merged category scores are then softmax-normalized.
Singleton categories (one form) simply skip the merge, so passing each form as its
own category yields a per-form distribution. The union of all forms is the closed
set that defines the normalization denominator.

The determiner study passes ``{"a": ["a", "an"], "the": ["the"]}``; this reproduces
the original ``_normalize_det_log_scores`` behavior bit-for-bit.

Scoring is adapted per architecture, exactly as in the source method:
- ``mlm``: read the mask-position logit for each form's token id.
- ``ar``: splice each form into the blank and score the full candidate sentence.
- ``seq2seq``: span-infilling via sentinel tokens, scored under teacher forcing.

Note: ``mlm`` currently resolves each form to a single token id; multi-token forms
need the (deferred) pseudo-log-likelihood path. ``ar``/``seq2seq`` already handle
multi-token forms at the string level.
"""
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.special import softmax

AR_BLANK_TOKEN = '___'


def get_default_device() -> torch.device:
    """Resolve the inference device: MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


_DEFAULT_DEVICE = get_default_device()


def _forms(candidates: Mapping[str, Sequence[str]]) -> list:
    """Ordered union of all surface forms across categories (closed candidate set)."""
    seen = []
    for forms in candidates.values():
        for f in forms:
            if f not in seen:
                seen.append(f)
    return seen


def merge_and_normalize(form_log_scores: Dict[str, float],
                        candidates: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    """Merge member forms per category (log-space) and softmax over categories.

    Bit-for-bit equivalent to ``_normalize_det_log_scores`` for the determiner
    candidate set ``{"a": ["a", "an"], "the": ["the"]}``.
    """
    cats = list(candidates.keys())
    cat_scores = []
    for forms in candidates.values():
        vals = [float(form_log_scores.get(f, -np.inf)) for f in forms]
        cat_scores.append(float(np.logaddexp.reduce(vals)) if len(vals) > 1 else float(vals[0]))
    arr = np.asarray(cat_scores, dtype=float)
    if not np.any(np.isfinite(arr)):
        return {c: 1.0 / len(cats) for c in cats}
    probs = softmax(arr)
    return {c: float(p) for c, p in zip(cats, probs)}


def _uniform(candidates: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    cats = list(candidates.keys())
    return {c: 1.0 / len(cats) for c in cats}


# ---------------------------------------------------------------------------
# Per-architecture scoring
# ---------------------------------------------------------------------------

def _score_mlm(model, tokenizer, prompt, candidates, device) -> Dict[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    mask_token_id = tokenizer.mask_token_id
    mask_position = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_position) == 0:
        return _uniform(candidates)
    mask_position = mask_position[0]

    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except TypeError as e:
            if "token_type_ids" in str(e) and "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
                outputs = model(**inputs)
            else:
                raise
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

    mask_logits = logits[0, mask_position, :].cpu().numpy()
    form_ids = {f: tokenizer.convert_tokens_to_ids(f) for f in _forms(candidates)}
    form_scores = {f: float(mask_logits[tid]) for f, tid in form_ids.items()}
    return merge_and_normalize(form_scores, candidates)


def _score_ar(model, tokenizer, prompt, candidates, device) -> Dict[str, float]:
    if prompt.find(AR_BLANK_TOKEN) == -1:
        return _uniform(candidates)

    scores = {}
    for form in _forms(candidates):
        full_sentence = prompt.replace(AR_BLANK_TOKEN, form)
        inputs = tokenizer(full_sentence, return_tensors="pt").to(device)
        if inputs['input_ids'].numel() == 0:
            scores[form] = -1000
            continue
        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except TypeError as e:
                if "token_type_ids" in str(e) and "token_type_ids" in inputs:
                    inputs.pop("token_type_ids")
                    outputs = model(**inputs)
                else:
                    raise
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        target_ids = inputs['input_ids'][0, 1:]
        token_logprobs = logprobs[range(len(target_ids)), target_ids]
        scores[form] = token_logprobs.sum().item()

    return merge_and_normalize(scores, candidates)


def _seq2seq_sentinel_tokens(tokenizer):
    """Resolve and validate span-infilling sentinels from the tokenizer."""
    cached = getattr(tokenizer, '_cached_seq2seq_sentinels', None)
    if cached is not None:
        return cached

    first, second = '<extra_id_0>', '<extra_id_1>'
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    for tok in (first, second):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None:
            raise ValueError(f"Tokenizer is missing required seq2seq sentinel token: {tok}")
        if isinstance(tok_id, int) and tok_id < 0:
            raise ValueError(f"Tokenizer returned invalid id for sentinel token {tok}: {tok_id}")
        if unk_id is not None and tok_id == unk_id:
            raise ValueError(
                f"Tokenizer maps required seq2seq sentinel token {tok} to unk_token_id ({unk_id}); "
                "cannot use seq2seq span-infilling path safely."
            )
    tokenizer._cached_seq2seq_sentinels = (first, second)
    return tokenizer._cached_seq2seq_sentinels


def _score_seq2seq_target(model, tokenizer, encoder_text, target_text, device) -> float:
    """Compute log P(target_text | encoder_text) under teacher forcing."""
    model_inputs = tokenizer(encoder_text, return_tensors='pt').to(device)
    label_ids = tokenizer(target_text, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        try:
            outputs = model(**model_inputs, labels=label_ids)
        except TypeError as e:
            if "token_type_ids" in str(e) and "token_type_ids" in model_inputs:
                model_inputs.pop("token_type_ids")
                outputs = model(**model_inputs, labels=label_ids)
            else:
                raise
        logits = outputs.logits[0]

    token_logprobs = torch.log_softmax(logits, dim=-1)
    targets = label_ids[0]
    valid = targets != tokenizer.pad_token_id
    if valid.sum().item() == 0:
        return float('-inf')
    gathered = token_logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(gathered[valid].sum().item())


def _score_seq2seq(model, tokenizer, prompt, candidates, device) -> Dict[str, float]:
    if prompt.find(AR_BLANK_TOKEN) == -1:
        return _uniform(candidates)
    s0, s1 = _seq2seq_sentinel_tokens(tokenizer)
    encoder_input = prompt.replace(AR_BLANK_TOKEN, s0)
    scores = {}
    for form in _forms(candidates):
        target = f"{s0} {form} {s1}"
        scores[form] = _score_seq2seq_target(model, tokenizer, encoder_input, target, device)
    return merge_and_normalize(scores, candidates)


_DISPATCH = {'mlm': _score_mlm, 'ar': _score_ar, 'seq2seq': _score_seq2seq}


def score_candidates(model, tokenizer, model_type: str, prompt: str,
                     candidates: Mapping[str, Sequence[str]],
                     device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return a normalized distribution over candidate *categories* at the blank.

    Args:
        model, tokenizer: a loaded model + tokenizer (see ``cac.load_model``).
        model_type: ``'mlm'`` | ``'ar'`` | ``'seq2seq'``.
        prompt: the assembled sentence containing the placeholder — ``[MASK]`` for
            MLM, ``___`` (``AR_BLANK_TOKEN``) for AR/seq2seq.
        candidates: ``{category_label: [surface_form, ...]}`` closed alternative set.
        device: inference device (defaults to MPS->CUDA->CPU).

    Returns:
        ``{category_label: probability}`` summing to 1.
    """
    if model_type not in _DISPATCH:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: mlm, ar, seq2seq")
    return _DISPATCH[model_type](model, tokenizer, prompt, candidates, device or _DEFAULT_DEVICE)
