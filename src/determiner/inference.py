"""Step 1: CAC inference for the determiner study (discourse condition, as in the paper).

For every D×N site, the determiner is replaced by a blank (or the model's mask token)
and the model sees all preceding utterances of the same recording session, labeled
``*CHI:`` / ``*MOT:``, trimmed from the start to fit its context window. The model's
preference between ``a`` (a/an) and ``the`` is stored as a probability distribution;
every downstream metric is computed analytically from those probabilities.
"""
import os
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import cac
from determiner.corpus import load_manchester_corpus
from determiner.sites import load_or_extract_det_noun_locations

# Closed candidate set: a/an are merged into the 'a' category.
DETERMINER_CANDIDATES = {'a': ['a', 'an'], 'the': ['the']}
BLANK_TOKEN = '___'           # placeholder for AR and seq2seq prompts
SPEAKER_LABELS = {'child': '*CHI', 'mother': '*MOT'}
CONTEXT_SAFETY_MARGIN = 10    # tokens kept free when packing the context window


def create_masked_input(sentence, det_char_start, det_char_end, model_type, mask_token=None):
    """Replace the determiner span with the mask token (MLM) or a ``___`` blank (AR, seq2seq)."""
    if model_type == 'mlm':
        placeholder = mask_token
    elif model_type in {'ar', 'seq2seq'}:
        placeholder = BLANK_TOKEN
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: mlm, ar, seq2seq")
    return sentence[:det_char_start] + placeholder + sentence[det_char_end:]


def prepare_discourse_inputs(corpus: Dict[str, pd.DataFrame], det_noun_locations: pd.DataFrame,
                             output_dir: str, model_configs_path: str) -> pd.DataFrame:
    """
    Build (or load) the model-agnostic discourse targets, cached as
    ``<output_dir>/discourse_cached_inputs.csv``.

    One row per site. ``context_start_line_num`` is the earliest prior line any evaluated
    model could attend to (-1 = no context); the context text itself is rebuilt from the
    corpus at inference time and trimmed per model.
    """
    cache_file = os.path.join(output_dir, 'discourse_cached_inputs.csv')
    required_cols = {
        'sentence_id', 'child_name', 'filename', 'line_num', 'speaker', 'noun',
        'original_sentence', 'original_det', 'masked_position',
        'det_char_start', 'det_char_end', 'context_start_line_num',
    }
    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file)
        if required_cols.issubset(cached_df.columns):
            print(f"  Loaded cached discourse inputs from {cache_file}")
            return cached_df
        print("  Cached discourse inputs are outdated; rebuilding...")

    print("  Computing discourse inputs for all sessions...")
    sample_df = pd.concat(list(corpus.values()), ignore_index=True)
    max_context_lines = cac.estimate_max_context_lines(model_configs_path, sample_df)
    targets = det_noun_locations.reset_index(drop=True)
    context_start = cac.compute_context_starts(corpus, targets, max_window_lines=max_context_lines)

    prepared_df = pd.DataFrame([{
        'sentence_id': i,
        'child_name': row['child_name'],
        'filename': row['filename'],
        'line_num': int(row['line_num']),
        'speaker': row['speaker_type'],
        'noun': row['noun'],
        'original_sentence': row['sentence'],
        'original_det': row['det'],
        'masked_position': int(row['det_token_idx']),
        'det_char_start': int(row['det_char_start']),
        'det_char_end': int(row['det_char_end']),
        'context_start_line_num': context_start.get(i, -1),
    } for i, row in targets.iterrows()])
    prepared_df.to_csv(cache_file, index=False)
    print(f"  Saved {len(prepared_df)} discourse inputs to {cache_file}")
    return prepared_df


def max_input_tokens(model, tokenizer) -> int:
    """Usable context length: the smaller of the tokenizer's and the model config's limits."""
    tokenizer_max = getattr(tokenizer, 'model_max_length', None)
    if tokenizer_max is None or tokenizer_max < 1 or tokenizer_max > 100000:
        tokenizer_max = 512  # tokenizers sometimes report a sentinel "infinite" length

    model_max = None
    cfg = getattr(model, 'config', None)
    for key in ('max_position_embeddings', 'n_positions', 'max_seq_len', 'max_sequence_length'):
        value = getattr(cfg, key, None)
        if value is not None and 1 <= int(value) <= 100000:
            model_max = int(value)
            break
    if model_max is None:
        model_max = tokenizer_max
    return min(int(tokenizer_max), int(model_max))


class _SessionLines:
    """One session's labeled utterance lines with cumulative token counts, for one model."""

    def __init__(self, session_df: pd.DataFrame, tokenizer):
        self.line_nums = session_df['line_num'].to_numpy(dtype=np.int64)
        self.lines = [
            f"{SPEAKER_LABELS['child'] if str(s).lower() == 'child' else SPEAKER_LABELS['mother']}: {sent}"
            for s, sent in zip(session_df['speaker_type'].astype(str), session_df['sentence'].astype(str))
        ]
        counts = np.array([len(tokenizer.encode(line, add_special_tokens=False)) for line in self.lines],
                          dtype=np.int64)
        self.prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        self.prefix[1:] = np.cumsum(counts)

    def context_before(self, target_line_num: int, token_budget: int) -> List[str]:
        """The longest run of whole lines immediately before the target that fits the budget."""
        end = int(np.searchsorted(self.line_nums, target_line_num, side='left'))
        lo, hi = 0, end
        while lo < hi:  # earliest start whose suffix fits the budget
            mid = (lo + hi) // 2
            if int(self.prefix[end] - self.prefix[mid]) <= token_budget:
                hi = mid
            else:
                lo = mid + 1
        return self.lines[lo:end]


def predict_speaker(model, tokenizer, model_type: str, model_name: str,
                    targets: pd.DataFrame, speaker_type: str,
                    session_lines: Dict, session_utterances: Dict) -> List[Dict]:
    """Score every site of one speaker (child or mother) for one model and dyad."""
    targets = targets[targets['speaker'] == speaker_type]
    if len(targets) == 0:
        print(f"  No determiner-noun pairs found for {speaker_type}")
        return []
    print(f"  Found {len(targets)} determiner-noun pairs for {speaker_type}")

    mask_token = tokenizer.mask_token if model_type == 'mlm' else None
    max_tokens = max_input_tokens(model, tokenizer)
    target_label = SPEAKER_LABELS[speaker_type]
    device = cac.get_default_device()

    predictions = []
    skipped_too_long = 0
    targets = targets.sort_values(['child_name', 'filename', 'line_num'])
    progress = tqdm(total=len(targets), desc=f"  Predicting {speaker_type}")
    for key, session_targets in targets.groupby(['child_name', 'filename'], sort=False):
        if key not in session_lines:
            session_df = session_utterances.get(key)
            session_lines[key] = (_SessionLines(session_df, tokenizer)
                                  if session_df is not None and len(session_df) else None)
        session = session_lines[key]

        for _, site in session_targets.iterrows():
            masked_sent = create_masked_input(site['original_sentence'], site['det_char_start'],
                                              site['det_char_end'], model_type, mask_token)
            target_line = f"{target_label}: {masked_sent}"
            budget = (max_tokens - len(tokenizer.encode(target_line, add_special_tokens=False))
                      - CONTEXT_SAFETY_MARGIN)

            context_lines, truncated = [], False
            if session is not None and budget > 0:
                context_lines = session.context_before(int(site['line_num']), budget)
                truncated = len(context_lines) < int(np.searchsorted(
                    session.line_nums, int(site['line_num']), side='left'))
                # Exact check with special tokens; drop whole lines from the start if needed.
                while context_lines:
                    candidate = "\n".join(context_lines + [target_line])
                    if len(tokenizer.encode(candidate, add_special_tokens=True)) <= max_tokens:
                        break
                    context_lines = context_lines[1:]
                    truncated = True

            prompt = "\n".join(context_lines + [target_line])
            if len(tokenizer.encode(prompt, add_special_tokens=True)) > max_tokens:
                skipped_too_long += 1  # the target alone does not fit; never cut an utterance
                progress.update(1)
                continue

            probs = cac.score_candidates(model, tokenizer, model_type, prompt,
                                         DETERMINER_CANDIDATES, device=device)
            predictions.append({
                'model_name': model_name,
                'model_type': model_type,
                'child_name': site['child_name'],
                'speaker': speaker_type,
                'sentence_id': int(site['sentence_id']),
                'original_sentence': site['original_sentence'],
                'masked_sentence': masked_sent,
                'noun': site['noun'],
                'masked_position': site['masked_position'],
                'original_det': site['original_det'],
                'P_a': probs['a'],
                'P_the': probs['the'],
                'predicted_det_argmax': 'a' if probs['a'] > probs['the'] else 'the',
                'context_length': len(context_lines),
                'full_input': prompt,
                'truncated': truncated,
            })
            progress.update(1)
    progress.close()

    if skipped_too_long:
        print(f"  Skipped {skipped_too_long} overlength inputs for {speaker_type}")
    return predictions


def save_predictions(predictions: List[Dict], output_dir: str, model_config: Dict,
                     child_name: str, speaker: str) -> str:
    """Write ``<output_dir>/<type>/<model>/<child>/<speaker>_predictions.csv``."""
    out_dir = os.path.join(output_dir, model_config['type'], model_config['name'].split('/')[-1], child_name)
    os.makedirs(out_dir, exist_ok=True)
    pred_file = os.path.join(out_dir, f"{speaker}_predictions.csv")
    pd.DataFrame(predictions).to_csv(pred_file, index=False)
    print(f"  Saved predictions to {pred_file}")
    return pred_file


def run_discourse_experiment(model_configs: List[Dict], corpus_path: str, output_dir: str,
                             model_configs_path: str, skip_existing: bool = True) -> None:
    """Run CAC inference for every model in ``model_configs`` over all Manchester sites."""
    corpus = load_manchester_corpus(corpus_path)
    det_noun_locations = load_or_extract_det_noun_locations(corpus, output_dir)
    prepared_inputs = prepare_discourse_inputs(corpus, det_noun_locations, output_dir, model_configs_path)
    print(f"Prepared {len(prepared_inputs)} discourse inputs")

    session_utterances = {
        (child_name, filename): grp.sort_values('line_num').reset_index(drop=True)
        for child_name, combined_df in corpus.items()
        for filename, grp in combined_df.groupby('filename', sort=False)
    }

    for config in model_configs:
        model_name, model_type = config['name'], config['type']
        print(f"\n{'=' * 80}\nProcessing {model_name} ({model_type.upper()})\n{'=' * 80}")
        try:
            _run_model(config, corpus, prepared_inputs, session_utterances, output_dir, skip_existing)
        except Exception:
            # Keep going: one failing model should not abort a multi-day run.
            print(f"ERROR processing {model_name}:")
            traceback.print_exc()


def _run_model(config: Dict, corpus: Dict, prepared_inputs: pd.DataFrame, session_utterances: Dict,
               output_dir: str, skip_existing: bool) -> None:
    model_type = config['type']
    base_model_name = config['name'].split('/')[-1]
    model, tokenizer = cac.load_model(config, revision=config.get('revision'))

    for child_name in corpus:
        out_dir = os.path.join(output_dir, model_type, base_model_name, child_name)
        pending = [spk for spk in ('child', 'mother')
                   if not (skip_existing and os.path.exists(os.path.join(out_dir, f"{spk}_predictions.csv")))]
        if not pending:
            print(f"  Skipping {child_name} (already processed)")
            continue

        print(f"\nProcessing {child_name}...")
        child_inputs = prepared_inputs[prepared_inputs['child_name'] == child_name]
        session_lines = {}  # per-session token counts, shared by the child and mother passes
        for speaker in pending:
            predictions = predict_speaker(model, tokenizer, model_type, base_model_name, child_inputs,
                                          speaker, session_lines, session_utterances)
            if predictions:
                save_predictions(predictions, output_dir, config, child_name, speaker)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
