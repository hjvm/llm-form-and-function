"""Discourse-context assembly for CAC — corpus/language-agnostic.

Two stages:
1. ``compute_context_starts`` walks each session once and records, per target site, the
   line-number boundary of the model-agnostic context window (capped at the largest model's
   window via ``estimate_max_context_lines``). Stored compactly as an int (no text).
2. At inference, reconstruct the utterances between that boundary and the target with
   ``reconstruct_context`` and render them with ``format_context_input``, trimming to each
   model's own token budget.

``format_context_input`` labels speakers via a ``speaker_label`` resolver (``callable(speaker)
-> str``). Default is passthrough on the corpus speaker column; a resolver returning a falsy
value emits the raw line (markerless). The determiner study passes a resolver that reproduces
``*CHI``/``*MOT`` byte-for-byte.
"""
import json
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd


def _max_length_from_name(name: str, mtype: str) -> int:
    n = name.lower()
    if mtype == 'ar' or any(k in n for k in ('gpt', 'llama', 'causal', 'opt')):
        return 1024
    return 512  # BERT / RoBERTa / T5


def estimate_max_context_lines(model_configs_path: str, corpus_sample: pd.DataFrame) -> int:
    """Maximum number of prior lines any model in ``model_configs_path`` can attend to.

    Reads tokenizer configs from the local HF cache (no weights), falls back to a name-based
    heuristic for uncached models, then divides the largest window by avg tokens-per-utterance
    in ``corpus_sample`` (needs a ``sentence`` column). Minimum 10.
    """
    from transformers import AutoTokenizer

    try:
        with open(model_configs_path) as f:
            configs = json.load(f)
    except Exception:
        configs = []

    max_model_tokens = 512
    for cfg in configs:
        name = cfg.get('name', '')
        mtype = cfg.get('type', 'mlm')
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            ml = getattr(tok, 'model_max_length', None)
            if ml and isinstance(ml, int) and 1 < ml <= 100_000:
                max_model_tokens = max(max_model_tokens, ml)
                continue
        except Exception:
            pass
        max_model_tokens = max(max_model_tokens, _max_length_from_name(name, mtype))

    sample_texts = corpus_sample['sentence'].dropna().head(2000).tolist()
    avg_tokens = max(1.0, sum(len(t) for t in sample_texts) / max(len(sample_texts), 1) / 4.0)
    max_lines = max(10, int(max_model_tokens / avg_tokens) + 5)
    print(f"  Context window estimate: {max_model_tokens} tokens, "
          f"avg {avg_tokens:.1f} tok/utterance → capping context at {max_lines} lines")
    return max_lines


def compute_context_starts(corpus: Dict[str, pd.DataFrame], target_sites: pd.DataFrame, *,
                           max_window_lines: int, child_key: str = 'child_name',
                           file_key: str = 'filename', line_col: str = 'line_num') -> Dict[int, int]:
    """Per-target context-window start (model-agnostic boundary), one linear pass per session.

    Returns ``{target_site_index: context_start_line_num}`` aligned to
    ``target_sites.reset_index(drop=True)``; ``-1`` means no prior context.
    """
    file_utterances = {}
    for child_name, combined_df in corpus.items():
        if len(combined_df) == 0:
            continue
        for filename, grp in combined_df.groupby(file_key, sort=False):
            file_utterances[(child_name, filename)] = grp.sort_values(line_col).reset_index(drop=True)

    targets = target_sites.reset_index(drop=True)
    context_start: Dict[int, int] = {}
    for (child_name, filename), group_targets in targets.groupby([child_key, file_key], sort=False):
        file_df = file_utterances.get((child_name, filename))
        if file_df is None:
            for idx in group_targets.index:
                context_start[idx] = -1
            continue
        file_line_nums = file_df[line_col].values
        n_file = len(file_df)
        sorted_targets = group_targets.sort_values(line_col)
        prior: deque = deque(maxlen=max_window_lines)
        file_idx = 0
        for idx, t_row in sorted_targets.iterrows():
            target_line = int(t_row[line_col])
            while file_idx < n_file and int(file_line_nums[file_idx]) < target_line:
                prior.append(int(file_line_nums[file_idx]))
                file_idx += 1
            context_start[idx] = int(prior[0]) if prior else -1
    return context_start


def reconstruct_context(session_df: pd.DataFrame, context_start_line_num: int,
                        target_line_num: int, line_col: str = 'line_num') -> List[dict]:
    """Utterances in ``[context_start_line_num, target_line_num)`` as ``{speaker_type, sentence}`` dicts."""
    if context_start_line_num is None or context_start_line_num < 0:
        return []
    window = session_df[(session_df[line_col] >= context_start_line_num)
                        & (session_df[line_col] < target_line_num)].sort_values(line_col)
    return [{'speaker_type': r.get('speaker_type'), 'sentence': r.get('sentence')}
            for _, r in window.iterrows()]


def format_context_input(context_utterances: Sequence, target_speaker: str, target_sentence: str,
                         model_type: str, *, speaker_label: Callable[[object], str] = str,
                         tokenizer=None, max_tokens: Optional[int] = None):
    """Render speaker-labeled context + masked target, trimmed to ``max_tokens``.

    ``speaker_label(speaker) -> str``: default passthrough on the speaker value; a falsy return
    emits the raw line (markerless). ``context_utterances`` may be ``list[dict]`` (with
    ``speaker_type``/``sentence``) or ``list[str]`` (legacy; assumed the non-target speaker).

    Returns ``(formatted_string, was_truncated)``.
    """
    def estimate_tokens(text: str) -> int:
        if tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text) // 4)

    context_lines = []
    for utt in context_utterances:
        if isinstance(utt, dict):
            lab = speaker_label(utt.get('speaker_type', 'mother'))
            sentence_text = str(utt.get('sentence', ''))
        else:
            lab = speaker_label('mother' if target_speaker == 'child' else 'child')
            sentence_text = str(utt)
        context_lines.append(f"{lab}: {sentence_text}" if lab else sentence_text)

    tlab = speaker_label(target_speaker)
    target_line = f"{tlab}: {target_sentence}" if tlab else target_sentence
    was_truncated = False

    if max_tokens is None or max_tokens <= 0:
        return "\n".join(context_lines + [target_line]), was_truncated

    base_tokens = estimate_tokens(target_line)
    available_tokens = max_tokens - base_tokens - 10  # safety margin

    included_context = []
    if available_tokens > 0:
        for line in reversed(context_lines):
            line_tokens = estimate_tokens(line + "\n")
            if available_tokens >= line_tokens:
                included_context.insert(0, line)
                available_tokens -= line_tokens
            else:
                was_truncated = True
                break
    elif len(context_lines) > 0:
        was_truncated = True

    if tokenizer is not None:
        while included_context:
            candidate = "\n".join(included_context + [target_line])
            try:
                tok_len = len(tokenizer.encode(candidate, add_special_tokens=True))
            except Exception:
                tok_len = len(candidate) // 4
            if tok_len <= max_tokens:
                break
            included_context = included_context[1:]
            was_truncated = True

    return "\n".join(included_context + [target_line]), was_truncated
