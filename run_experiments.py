#!/usr/bin/env python3
"""
Run syntactic productivity experiments on language models.

Supports three experiment types:
1. Isolated: Process single sentences without discourse context
2. Discourse: Process sentences with multi-turn discourse context
3. Contextual: Use TPR-style context (prior mention by OTHER speaker) for ALL pairs

Usage:
    # Run isolated experiment
    python run_experiments.py isolated --models model_configs.json

    # Run discourse experiment (main experiment for the paper; CHILDES *CHI/*MOT labels by default)
    python run_experiments.py discourse --models model_configs.json
    # ...or override the speaker labels: --discourse-label-style spoken

    # Run contextual experiment (TPR-style context, all det-noun pairs)
    python run_experiments.py contextual --models model_configs.json

    # Run both isolated and discourse experiments
    python run_experiments.py both --models model_configs.json

    # Run all experiments (isolated + discourse + contextual)
    python run_experiments.py all --models model_configs.json

    # Force reprocessing (overwrite existing results)
    python run_experiments.py discourse --models model_configs.json --force
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from cac_utils import *
import numpy as np

# Default paths
DEFAULT_CORPUS_PATH = './data/CHILDES/Manchester'
DEFAULT_OUTPUT_PATH_ISOLATED = './output/manchester_masked'
DEFAULT_OUTPUT_PATH_DISCOURSE = './output/manchester_discourse'
DEFAULT_OUTPUT_PATH_CONTEXTUAL = './output/manchester_contextual'


def _style_output_path(base_path: str, discourse_label_style: str) -> str:
    if discourse_label_style == 'childes' and not base_path.endswith('_childes'):
        return f"{base_path}_childes"
    return base_path


def run_isolated_experiment(model_configs, corpus_path, output_dir, skip_existing=True):
    """
    Run the isolated utterance experiment.
    
    Processes each model on single sentences without discourse context.
    """
    print("\n" + "="*80)
    print("RUNNING ISOLATED UTTERANCE EXPERIMENT")
    print("="*80 + "\n")
    
    # Load corpus
    print("Loading Manchester corpus...")
    corpus = load_manchester_corpus(corpus_path, cache_dir=output_dir)
    
    # Load or extract det-noun locations (done once, reused for all models)
    det_noun_locations = load_or_extract_det_noun_locations(corpus, output_dir)
    
    # Process each model
    for config in model_configs:
        model_name = config['name']
        model_type = config['type']
        
        print(f"\n{'='*80}")
        print(f"Processing {model_name} ({model_type.upper()})")
        print(f"{'='*80}\n")
        
        try:
            # Load model
            model, tokenizer = load_model(config)
            
            # Get determiner token IDs
            det_token_ids = {det: tokenizer.convert_tokens_to_ids(det) 
                           for det in DETERMINERS}
            
            # Process each child
            for child_name, combined_df in corpus.items():
                print(f"\nProcessing {child_name}...")
                
                # Check if already processed (skip_existing logic)
                base_model_name = model_name.split('/')[-1]
                out_dir = f"{output_dir}/{model_type}/{base_model_name}/{child_name}"
                
                child_pred_file = f"{out_dir}/child_predictions.csv"
                mother_pred_file = f"{out_dir}/mother_predictions.csv"
                
                # Extract child and mother utterances
                child_df = combined_df[combined_df['speaker_type'] == 'child'][['speaker', 'sentence']]
                mother_df = combined_df[combined_df['speaker_type'] == 'mother'][['speaker', 'sentence']]
                
                # Process child utterances (skip if exists)
                if skip_existing and os.path.exists(child_pred_file):
                    print(f"  ✓ Skipping child (already processed)")
                else:
                    child_predictions = process_speaker_isolated(
                        model, tokenizer, model_type, det_token_ids, det_noun_locations,
                        child_name, 'child', base_model_name
                    )
                    if child_predictions:
                        save_predictions(child_predictions, output_dir, config, child_name, 'child')
                
                # Process mother utterances (skip if exists)
                if skip_existing and os.path.exists(mother_pred_file):
                    print(f"  ✓ Skipping mother (already processed)")
                else:
                    mother_predictions = process_speaker_isolated(
                        model, tokenizer, model_type, det_token_ids, det_noun_locations,
                        child_name, 'mother', base_model_name
                    )
                    if mother_predictions:
                        save_predictions(mother_predictions, output_dir, config, child_name, 'mother')
            
            # Generate summary statistics for this model
            print(f"\nGenerating summaries for {model_name}...")
            try:
                generate_model_summaries(output_dir, config)
            except Exception as e:
                print(f"  Warning: Could not generate summaries: {e}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ISOLATED EXPERIMENT COMPLETE")
    print("="*80 + "\n")


def process_speaker_isolated(model, tokenizer, model_type, det_token_ids, det_noun_locations,
                             child_name, speaker_type, model_name):
    """Process utterances for a single speaker in isolated mode using pre-extracted locations."""
    from tqdm import tqdm
    
    predictions = []
    
    # Filter pre-extracted locations for this child and speaker
    speaker_locations = det_noun_locations[
        (det_noun_locations['child_name'] == child_name) &
        (det_noun_locations['speaker_type'] == speaker_type)
    ]
    
    if len(speaker_locations) == 0:
        print(f"  No determiner-noun pairs found for {speaker_type}")
        return None
    
    print(f"  Found {len(speaker_locations)} determiner-noun pairs for {speaker_type}")
    
    # Get mask token for MLM models
    mask_token = tokenizer.mask_token if model_type == 'mlm' else None
    
    # Process each location
    for idx, loc in tqdm(speaker_locations.iterrows(), total=len(speaker_locations), desc=f"  Predicting {speaker_type}"):
        # Create masked input
        masked_sent = create_masked_input(
            loc['sentence'],
            loc['det_char_start'],
            loc['det_char_end'],
            model_type,
            mask_token
        )
        
        # Get predictions (shared dispatch for mlm/ar/seq2seq).
        probs = extract_determiner_probabilities(model, tokenizer, model_type, masked_sent, det_token_ids)

        # Deterministic and probabilistic choices.
        det_argmax, det_sample = choose_determiner_predictions(probs)
        
        # Match original notebook column names EXACTLY
        predictions.append({
            'model_name': model_name,
            'model_type': model_type,
            'child_name': child_name,
            'speaker': speaker_type,
            'sentence_id': idx,  # Use DataFrame index as sentence_id
            'original_sentence': loc['sentence'],
            'masked_sentence': masked_sent,
            'noun': loc['noun'],
            'masked_position': loc['det_token_idx'],
            'original_det': loc['det'],
            'P_a': probs['a'],
            'P_the': probs['the'],
            'predicted_det_argmax': det_argmax,
            'predicted_det_sample': det_sample,
        })
    
    return predictions


def run_discourse_experiment(model_configs, corpus_path, output_dir, skip_existing=True,
                             discourse_label_style='spoken'):
    """
    Run the discourse context experiment.
    
    Processes each model on sentences with multi-turn discourse context.
    """
    print("\n" + "="*80)
    print("RUNNING DISCOURSE CONTEXT EXPERIMENT")
    print("="*80 + "\n")
    print(f"Discourse label style: {discourse_label_style}")
    
    # Load corpus
    print("Loading Manchester corpus...")
    corpus = load_manchester_corpus(corpus_path)
    
    # Load or extract det-noun locations (done once, reused for all models)
    det_noun_locations = load_or_extract_det_noun_locations(corpus, output_dir)

    # Pre-compute model-agnostic discourse inputs once (reused across all models)
    print("\nPre-computing discourse cached inputs (all prior utterances)...")
    prepared_inputs = prepare_discourse_inputs_all_sessions(
        corpus, det_noun_locations, output_dir, model_configs_path='model_configs.json',
        discourse_label_style=discourse_label_style
    )
    print(f"✓ Prepared {len(prepared_inputs)} cached discourse inputs")

    # Build per-file corpus lookup used at inference time to reconstruct context
    # from the compact (context_start_line_num, line_num) pair in the cache.
    file_utterances: dict = {}
    for child_name, combined_df in corpus.items():
        for filename, grp in combined_df.groupby('filename', sort=False):
            file_utterances[(child_name, filename)] = (
                grp.sort_values('line_num').reset_index(drop=True)
            )
    
    # Process each model
    for config in model_configs:
        model_name = config['name']
        model_type = config['type']
        
        print(f"\n{'='*80}")
        print(f"Processing {model_name} ({model_type.upper()})")
        print(f"{'='*80}\n")
        
        try:
            # Load model
            model, tokenizer = load_model(config)
            
            # Get determiner token IDs
            det_token_ids = {det: tokenizer.convert_tokens_to_ids(det) 
                           for det in DETERMINERS}
            
            # Process each child
            for child_name, combined_df in corpus.items():
                print(f"\nProcessing {child_name}...")
                
                # Check if already processed BEFORE any heavy processing
                base_model_name = model_name.split('/')[-1]
                out_dir = f"{output_dir}/{model_type}/{base_model_name}/{child_name}"
                
                child_pred_file = f"{out_dir}/child_predictions.csv"
                mother_pred_file = f"{out_dir}/mother_predictions.csv"
                
                # Skip this child entirely if both speakers are already done
                if skip_existing and os.path.exists(child_pred_file) and os.path.exists(mother_pred_file):
                    print(f"  ✓ Skipping {child_name} (both child and mother already processed)")
                    continue
                
                # Filter pre-computed cached inputs for this child
                child_inputs = prepared_inputs[prepared_inputs['child_name'] == child_name]
                # Shared per-file token cache for both child and mother passes.
                token_window_cache = {}
                
                # Process child utterances (skip if exists)
                if skip_existing and os.path.exists(child_pred_file):
                    print(f"  ✓ Skipping child (already processed)")
                else:
                    child_speaker_inputs = child_inputs[child_inputs['speaker'] == 'child']
                    child_predictions = process_speaker_discourse(
                        model, tokenizer, model_type, det_token_ids,
                        child_speaker_inputs, 'child', base_model_name,
                        file_utterances=file_utterances,
                        file_token_cache=token_window_cache,
                        discourse_label_style=discourse_label_style
                    )
                    if child_predictions:
                        save_predictions(child_predictions, output_dir, config, child_name, 'child')
                
                # Process mother utterances (skip if exists)
                if skip_existing and os.path.exists(mother_pred_file):
                    print(f"  ✓ Skipping mother (already processed)")
                else:
                    mother_speaker_inputs = child_inputs[child_inputs['speaker'] == 'mother']
                    mother_predictions = process_speaker_discourse(
                        model, tokenizer, model_type, det_token_ids,
                        mother_speaker_inputs, 'mother', base_model_name,
                        file_utterances=file_utterances,
                        file_token_cache=token_window_cache,
                        discourse_label_style=discourse_label_style
                    )
                    if mother_predictions:
                        save_predictions(mother_predictions, output_dir, config, child_name, 'mother')
            
            # Generate summary statistics for this model
            print(f"\nGenerating summaries for {model_name}...")
            try:
                generate_model_summaries(output_dir, config)
            except Exception as e:
                print(f"  Warning: Could not generate summaries: {e}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("DISCOURSE EXPERIMENT COMPLETE")
    print("="*80 + "\n")


def process_speaker_discourse(model, tokenizer, model_type, det_token_ids,
                              prepared_inputs_df, speaker_type, model_name,
                              file_utterances=None, file_token_cache=None,
                              discourse_label_style='spoken'):
    """Process one speaker in discourse mode from cached model-agnostic inputs."""
    from tqdm import tqdm
    
    predictions = []
    skipped_too_long = 0
    
    # Filter to this speaker from prepared cached inputs
    speaker_locations = prepared_inputs_df[prepared_inputs_df['speaker'] == speaker_type]

    if len(speaker_locations) == 0:
        print(f"  No determiner-noun pairs found for {speaker_type}")
        return None
    
    print(f"  Found {len(speaker_locations)} determiner-noun pairs for {speaker_type}")
    
    # Get mask token for MLM models
    mask_token = tokenizer.mask_token if model_type == 'mlm' else None

    # Derive a safe effective context cap from BOTH tokenizer and model config.
    # Tokenizers sometimes report sentinel max values that exceed model capacity.
    tokenizer_max_tokens = getattr(tokenizer, 'model_max_length', None)
    if tokenizer_max_tokens is None or tokenizer_max_tokens < 1 or tokenizer_max_tokens > 100000:
        tokenizer_max_tokens = 512

    cfg = getattr(model, 'config', None)
    model_config_max_tokens = None
    for key in ('max_position_embeddings', 'n_positions', 'max_seq_len', 'max_sequence_length'):
        value = getattr(cfg, key, None)
        if value is not None and 1 <= int(value) <= 100000:
            model_config_max_tokens = int(value)
            break
    if model_config_max_tokens is None:
        model_config_max_tokens = tokenizer_max_tokens

    effective_max_tokens = min(int(tokenizer_max_tokens), int(model_config_max_tokens))
    if effective_max_tokens != tokenizer_max_tokens:
        print(
            f"  Applying model positional cap: tokenizer_max={tokenizer_max_tokens}, "
            f"model_max={model_config_max_tokens}, effective_max={effective_max_tokens}"
        )
    
    # Sort by file and chronology so each file can be handled in one forward pass.
    speaker_locations = speaker_locations.sort_values(['child_name', 'filename', 'line_num'])

    def _label_for_speaker(s):
        speaker = str(s).lower()
        if discourse_label_style == 'childes':
            return '*CHI' if speaker == 'child' else '*MOT'
        return 'CHILD' if speaker == 'child' else 'MOTHER'

    def _get_file_runtime_cache(child_name, filename):
        if file_token_cache is None:
            return None
        key = (child_name, filename)
        if key in file_token_cache:
            return file_token_cache[key]
        if file_utterances is None:
            return None

        file_df = file_utterances.get(key)
        if file_df is None or len(file_df) == 0:
            file_token_cache[key] = None
            return None

        line_nums = file_df['line_num'].to_numpy(dtype=np.int64)
        lines = [
            f"{_label_for_speaker(s)}: {sent}"
            for s, sent in zip(file_df['speaker_type'].astype(str).tolist(),
                               file_df['sentence'].astype(str).tolist())
        ]
        # Tokenize each utterance line once for this model.
        line_tok_counts = np.array([
            len(tokenizer.encode(line, add_special_tokens=False))
            for line in lines
        ], dtype=np.int64)
        prefix = np.zeros(len(line_tok_counts) + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(line_tok_counts)

        payload = {
            'line_nums': line_nums,
            'lines': lines,
            'prefix': prefix,
        }
        file_token_cache[key] = payload
        return payload

    progress = tqdm(total=len(speaker_locations), desc=f"  Predicting {speaker_type}")
    for (child_name, filename), file_targets in speaker_locations.groupby(['child_name', 'filename'], sort=False):
        runtime = _get_file_runtime_cache(child_name, filename)

        for _, loc in file_targets.iterrows():
        # Create masked input using the SAME function as isolated experiment
            masked_sent = create_masked_input(
                loc['original_sentence'],
                loc['det_char_start'],
                loc['det_char_end'],
                model_type,
                mask_token
            )

            if discourse_label_style == 'childes':
                target_label = '*CHI' if speaker_type == 'child' else '*MOT'
            else:
                target_label = 'CHILD' if speaker_type == 'child' else 'MOTHER'
            target_line = f"{target_label}: {masked_sent}"
            target_tokens = len(tokenizer.encode(target_line, add_special_tokens=False))
            available_tokens = effective_max_tokens - target_tokens - 10  # safety margin

            context_lines = []
            was_truncated = False

            if runtime is not None and available_tokens > 0:
                line_nums = runtime['line_nums']
                lines = runtime['lines']
                prefix = runtime['prefix']

                t_line = int(loc.get('line_num', -1))
                end_idx = int(np.searchsorted(line_nums, t_line, side='left'))
                if end_idx > 0:
                    # Find earliest start index whose suffix token sum fits budget.
                    lo, hi = 0, end_idx
                    while lo < hi:
                        mid = (lo + hi) // 2
                        tok_sum = int(prefix[end_idx] - prefix[mid])
                        if tok_sum <= available_tokens:
                            hi = mid
                        else:
                            lo = mid + 1
                    start_idx = lo
                    context_lines = lines[start_idx:end_idx]
                    was_truncated = start_idx > 0

                    # Exact check and whole-line fallback to avoid any overflow.
                    if context_lines:
                        while context_lines:
                            candidate = "\n".join(context_lines + [target_line])
                            tok_len = len(tokenizer.encode(candidate, add_special_tokens=True))
                            if tok_len <= effective_max_tokens:
                                break
                            context_lines = context_lines[1:]
                            was_truncated = True

            discourse_input = "\n".join(context_lines + [target_line])
            context_length = len(context_lines)

            # If target + context still exceeds model limit, skip this example.
            # We preserve whole utterances, so we do not partially trim target text.
            final_tokens = len(tokenizer.encode(discourse_input, add_special_tokens=True))
            if final_tokens > effective_max_tokens:
                skipped_too_long += 1
                progress.update(1)
                continue
        
            # Get predictions using the shared model-family dispatch.
            probs = extract_determiner_probabilities(model, tokenizer, model_type, discourse_input, det_token_ids)

            # Deterministic and probabilistic choices.
            det_argmax, det_sample = choose_determiner_predictions(probs)
        
            # Match original notebook column names EXACTLY (discourse has extra columns)
            predictions.append({
                'model_name': model_name,
                'model_type': model_type,
                'child_name': loc['child_name'],
                'speaker': speaker_type,
                'sentence_id': int(loc['sentence_id']),
                'original_sentence': loc['original_sentence'],
                'masked_sentence': masked_sent,
                'noun': loc['noun'],
                'masked_position': loc['masked_position'],
                'original_det': loc['original_det'],
                'P_a': probs['a'],
                'P_the': probs['the'],
                'predicted_det_argmax': det_argmax,
                'predicted_det_sample': det_sample,
                'context_length': context_length,
                'full_input': discourse_input,
                'truncated': was_truncated,
            })
            progress.update(1)

    progress.close()
    if skipped_too_long > 0:
        print(f"  Skipped {skipped_too_long} overlength inputs for {speaker_type}")

    return predictions


def run_contextual_experiment(model_configs, corpus_path, output_dir, skip_existing=True):
    """
    Run the contextual discourse experiment.
    
    Uses TPR-style cross-speaker context for ALL det-noun pairs:
    - Processes ALL nouns (no attested filter like TPR)
    - ONLY includes cases where OTHER speaker mentioned that noun before
    - Every input includes that prior mention + all intervening utterances
    
    This gives more observations than TPR (no attested filter) while maintaining
    the same cross-speaker discourse context requirement.
    
    Args:
        model_configs: List of model configuration dictionaries
        corpus_path: Path to Manchester corpus directory
        output_dir: Output directory
        skip_existing: Skip already processed files (default: True)
    """
    print("\n" + "="*80)
    print("RUNNING CONTEXTUAL DISCOURSE EXPERIMENT")
    print("="*80 + "\n")
    print("This uses TPR-style context (OTHER speaker's prior mention + intervening)")
    print("for ALL nouns (no attested filter). All inputs include cross-speaker context.\n")
    
    # Load corpus
    print("Loading Manchester corpus...")
    corpus = load_manchester_corpus(corpus_path, cache_dir=output_dir)
    
    # Load or extract det-noun locations (done once, reused for all models)
    det_noun_locations = load_or_extract_det_noun_locations(corpus, output_dir)
    
    # Pre-compute contextual inputs (cached once, reused for all models)
    print("\nPre-computing contextual inputs (cases with OTHER speaker's prior mention)...")
    prepared_inputs = prepare_contextual_inputs_all_sessions(det_noun_locations, output_dir)
    
    print(f"\n✓ Prepared {len(prepared_inputs)} contextual inputs (all with cross-speaker context)")
    
    # Process each model
    for config in model_configs:
        model_name = config['name']
        model_type = config['type']
        base_model_name = model_name.split('/')[-1]
        
        print(f"\n{'='*80}")
        print(f"Processing {model_name} ({model_type.upper()})")
        print(f"{'='*80}\n")
        
        try:
            # Load model
            model, tokenizer = load_model(config)
            det_token_ids = {det: tokenizer.convert_tokens_to_ids(det) 
                           for det in DETERMINERS}
            
            # Process each child
            for child_name in prepared_inputs['child_name'].unique():
                print(f"\nProcessing {child_name}...")
                
                # Check if already processed
                out_dir = f"{output_dir}/{model_type}/{base_model_name}/{child_name}"
                child_pred_file = f"{out_dir}/child_predictions.csv"
                mother_pred_file = f"{out_dir}/mother_predictions.csv"
                
                # Skip if both already exist
                if skip_existing and os.path.exists(child_pred_file) and os.path.exists(mother_pred_file):
                    print(f"  ✓ Skipping {child_name} (both child and mother already processed)")
                    continue
                
                # Filter inputs for this child
                child_inputs = prepared_inputs[prepared_inputs['child_name'] == child_name]
                
                # Process child utterances
                if skip_existing and os.path.exists(child_pred_file):
                    print(f"  ✓ Skipping child (already processed)")
                else:
                    child_speaker_inputs = child_inputs[child_inputs['speaker'] == 'child']
                    if len(child_speaker_inputs) > 0:
                        child_predictions = process_speaker_contextual(
                            model, tokenizer, model_type, det_token_ids,
                            child_speaker_inputs, base_model_name
                        )
                        if child_predictions:
                            save_predictions(child_predictions, output_dir, config, child_name, 'child')
                
                # Process mother utterances
                if skip_existing and os.path.exists(mother_pred_file):
                    print(f"  ✓ Skipping mother (already processed)")
                else:
                    mother_speaker_inputs = child_inputs[child_inputs['speaker'] == 'mother']
                    if len(mother_speaker_inputs) > 0:
                        mother_predictions = process_speaker_contextual(
                            model, tokenizer, model_type, det_token_ids,
                            mother_speaker_inputs, base_model_name
                        )
                        if mother_predictions:
                            save_predictions(mother_predictions, output_dir, config, child_name, 'mother')
            
            # Generate summary statistics for this model
            print(f"\nGenerating summaries for {model_name}...")
            try:
                generate_model_summaries(output_dir, config)
            except Exception as e:
                print(f"  Warning: Could not generate summaries: {e}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("CONTEXTUAL EXPERIMENT COMPLETE")
    print("="*80 + "\n")


def process_speaker_contextual(model, tokenizer, model_type, det_token_ids, prepared_inputs_df, model_name):
    """
    Generate predictions for one speaker using prepared contextual inputs.
    
    All inputs include cross-speaker context (OTHER speaker's prior mention + intervening).
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        model_type: 'mlm', 'ar', or 'seq2seq'
        det_token_ids: Dict mapping determiner -> token_id
        prepared_inputs_df: DataFrame with pre-formatted inputs (all with context)
        model_name: Name of the model
        
    Returns:
        List of prediction dicts
    """
    from tqdm import tqdm
    
    predictions = []
    
    # Process each prepared input (all have cross-speaker context)
    for idx, row in tqdm(prepared_inputs_df.iterrows(), total=len(prepared_inputs_df), desc="  Predicting"):
        # Get contextual input with OTHER speaker's prior mention
        if model_type == 'mlm':
            # Replace generic {MASK} with model-specific mask token
            full_input = row['full_input_mlm'].replace('{MASK}', tokenizer.mask_token)
        else:
            full_input = row['full_input_ar']
        
        context_length = 1  # At minimum, OTHER speaker's prior mention
        
        # Get predictions (shared dispatch for mlm/ar/seq2seq).
        probs = extract_determiner_probabilities(model, tokenizer, model_type, full_input, det_token_ids)

        # Deterministic and probabilistic choices.
        det_argmax, det_sample = choose_determiner_predictions(probs)
        
        # Match TPR/discourse experiment columns
        predictions.append({
            'model_name': model_name,
            'model_type': model_type,
            'child_name': row['child_name'],
            'speaker': row['speaker'],
            'sentence_id': row['sentence_id'],
            'original_sentence': row['original_sentence'],
            'noun': row['noun'],
            'masked_position': row['masked_position'],
            'original_det': row['original_det'],
            'previous_det': row['previous_det'],
            'previous_speaker': row['previous_speaker'],
            'P_a': probs['a'],
            'P_the': probs['the'],
            'predicted_det_argmax': det_argmax,
            'predicted_det_sample': det_sample,
            'context_length': context_length,
            'full_input': full_input,
            'truncated': False,
        })
    
    return predictions



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run LM syntactic productivity experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('experiment',
                       choices=['isolated', 'discourse', 'contextual', 'both', 'all'],
                       help='Which experiment to run (both = isolated + discourse; all = isolated + discourse + contextual)')
    parser.add_argument('--models', 
                       required=True, 
                       help='Path to model config JSON file')
    parser.add_argument('--corpus', 
                       default=DEFAULT_CORPUS_PATH, 
                       help=f'Corpus path (default: {DEFAULT_CORPUS_PATH})')
    parser.add_argument('--output-isolated', 
                       default=DEFAULT_OUTPUT_PATH_ISOLATED, 
                       help=f'Output directory for isolated experiment (default: {DEFAULT_OUTPUT_PATH_ISOLATED})')
    parser.add_argument('--output-discourse', 
                       default=DEFAULT_OUTPUT_PATH_DISCOURSE,
                       help=f'Output directory for discourse experiment (default: {DEFAULT_OUTPUT_PATH_DISCOURSE})')
    parser.add_argument('--output-contextual',
                       default=DEFAULT_OUTPUT_PATH_CONTEXTUAL,
                       help=f'Output directory for contextual experiment (default: {DEFAULT_OUTPUT_PATH_CONTEXTUAL})')
    parser.add_argument('--force',
                       action='store_true', 
                       help='Overwrite existing results')
    parser.add_argument('--discourse-label-style',
                       choices=['spoken', 'childes'],
                       default='childes',
                       help="Speaker label style for discourse prompts (default: childes = *CHI/*MOT, "
                            "as used in the paper; pass 'spoken' for MOTHER/CHILD labels)")
    
    args = parser.parse_args()
    
    # Load model configs
    try:
        with open(args.models) as f:
            model_configs = json.load(f)
        print(f"Loaded {len(model_configs)} model configurations")
    except Exception as e:
        print(f"ERROR: Could not load model configs from {args.models}: {e}")
        sys.exit(1)
    
    # Run requested experiment(s)
    if args.experiment in ['isolated', 'both', 'all']:
        run_isolated_experiment(
            model_configs, 
            args.corpus, 
            args.output_isolated, 
            skip_existing=not args.force
        )
    
    if args.experiment in ['discourse', 'both', 'all']:
        run_discourse_experiment(
            model_configs, 
            args.corpus, 
            _style_output_path(args.output_discourse, args.discourse_label_style),
            skip_existing=not args.force,
            discourse_label_style=args.discourse_label_style
        )
    
    if args.experiment in ['contextual', 'all']:
        run_contextual_experiment(
            model_configs,
            args.corpus,
            args.output_contextual,
            skip_existing=not args.force
        )

    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()
