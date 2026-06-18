"""
Utility functions for language model syntactic productivity experiments.

This module contains core functionality for:
- Loading CHILDES Manchester corpus
- Extracting determiner probabilities from MLM and AR models
- Building discourse context
- Processing predictions and calculating overlap metrics
- Saving and loading experimental results

Usage:
    from cac_utils import *
    
    # Load corpus
    corpus = load_manchester_corpus('./data/CHILDES')
    
    # Process model
    model, tokenizer = load_model(model_config)
    predictions = process_model_isolated(model, tokenizer, corpus, extractor)
"""

import json
import os
import re
import time
from collections import deque
from datetime import date
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import ttest_rel
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

import cac

# Import pair extractor
try:
    from determiner.pair_extractors import DeterminerNounExtractor
except ImportError:
    print("Warning: pair_extractors.py not found. DeterminerNounExtractor will not be available.")
    DeterminerNounExtractor = None


# =============================================================================
# CONSTANTS
# =============================================================================

DETERMINERS = ['a', 'an', 'the']
DETERMINER_CANDIDATES = {'a': ['a', 'an'], 'the': ['the']}  # a/an merged into the 'a' category
AR_BLANK_TOKEN = '___'
MLM_BLANK_TOKEN = '[MASK]'
MAX_SKIP = 3

# Default paths
DEFAULT_CORPUS_PATH = './data/CHILDES/Manchester'
DEFAULT_OUTPUT_PATH_ISOLATED = './output/manchester_masked'
DEFAULT_OUTPUT_PATH_DISCOURSE = './output/manchester_discourse'

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# =============================================================================
# DETERMINER-NOUN PAIR EXTRACTION AND CACHING
# =============================================================================

def extract_and_save_det_noun_locations(corpus, output_dir):
    """
    Extract all determiner-noun pair locations from the corpus and save to CSV.
    
    This should be called once per output directory. The extracted locations are
    saved and reused across all models and experiments to ensure consistency.
    
    Extracts using spaCy to get exact character positions needed for masking.
    
    Args:
        corpus: Dict[child_name: DataFrame] from load_manchester_corpus
        output_dir: Directory where det_noun_locations.csv will be saved
        
    Returns:
        DataFrame with columns: child_name, filename, line_num, speaker, speaker_type,
                                sentence, det, noun, pair_index, det_char_start, det_char_end
    """
    if DeterminerNounExtractor is None:
        raise RuntimeError("DeterminerNounExtractor not available. Install spacy: pip install spacy && python -m spacy download en_core_web_sm")
    
    print(f"Extracting determiner-noun pairs from corpus...")
    extractor = DeterminerNounExtractor('en_core_web_sm')
    
    all_locations = []
    
    for child_name, child_df in corpus.items():
        print(f"  Processing {child_name}...")
        
        # Process all sentences in batches for efficiency
        sentences = child_df['sentence'].tolist()
        docs = extractor.nlp.pipe(sentences, batch_size=extractor.pipe_batch_size)
        
        for sent_idx, (doc, (idx, row)) in enumerate(zip(docs, child_df.iterrows())):
            sentence = row['sentence']
            
            # Extract pairs using spaCy to get character positions
            try:
                pair_idx = 0
                for chunk in doc.noun_chunks:
                    if chunk[0].pos_ == "DET" and chunk[0].lower_ in ("a", "an", "the"):
                        det_token = chunk[0]
                        det = "a" if det_token.lower_ in ("a", "an") else "the"
                        
                        det_idx = chunk[0].i
                        noun_idx = chunk.root.i
                        gap = noun_idx - det_idx - 1
                        
                        if 0 <= gap <= extractor.max_gap:
                            if chunk.root.pos_ == "NOUN" and "Sing" in chunk.root.morph.get("Number"):
                                all_locations.append({
                                    'child_name': child_name,
                                    'filename': row['filename'],
                                    'line_num': row['line_num'],
                                    'speaker': row['speaker'],
                                    'speaker_type': row['speaker_type'],
                                    'sentence': sentence,
                                    'det': det.lower(),
                                    'noun': chunk.root.lemma_.lower(),
                                    'pair_index': pair_idx,
                                    'det_char_start': det_token.idx,
                                    'det_char_end': det_token.idx + len(det_token.text),
                                    'det_token_idx': det_idx,
                                    'noun_token_idx': noun_idx
                                })
                                pair_idx += 1
            except Exception as e:
                continue
    
    locations_df = pd.DataFrame(all_locations)
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'det_noun_locations.csv')
    locations_df.to_csv(output_file, index=False)
    print(f"Saved {len(locations_df)} determiner-noun pairs to {output_file}")
    
    return locations_df


def load_or_extract_det_noun_locations(corpus, output_dir):
    """
    Load pre-extracted det-noun locations if available, otherwise extract and save.
    
    This ensures we only extract pairs once per output directory, maintaining
    consistency across all models and experiments.
    
    Args:
        corpus: Dict[child_name: DataFrame] from load_manchester_corpus
        output_dir: Directory where det_noun_locations.csv is stored
        
    Returns:
        DataFrame with det-noun pair locations
    """
    locations_file = os.path.join(output_dir, 'det_noun_locations.csv')
    
    if os.path.exists(locations_file):
        print(f"Loading pre-extracted det-noun pairs from {locations_file}")
        locations_df = pd.read_csv(locations_file)
        print(f"Loaded {len(locations_df)} determiner-noun pairs")
        return locations_df
    else:
        print(f"No pre-extracted pairs found. Extracting from corpus...")
        return extract_and_save_det_noun_locations(corpus, output_dir)


# =============================================================================
# CORPUS LOADING AND PREPROCESSING
# =============================================================================

def clean_childes_annotations(text: str) -> str:
    """
    Remove CHILDES annotation markers from transcript text.
    
    This cleans the following markers (based on CHILDES CHAT manual):
    - xxx: unidentifiable speech material
    - 0word: omitted words (e.g., "0to", "0is")
    - word+word: compounds (keep as single token without +)
    - &=action: simple events (e.g., &=coughs, &=laughs)
    - &-filler: filled pauses (e.g., &-um, &-uh)
    - [text]: annotations in brackets (reformulations, paralinguistics, etc.)
    - <text>: angle brackets for repeated/revised material
    - Special symbols: +..., +/., +//, +^, etc.
    - Phonological: xxx, yyy, www
    - Other non-word strings: &+fragment, &word
    
    Args:
        text: Raw CHILDES transcript text
        
    Returns:
        Cleaned text suitable for language model input
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Remove speaker codes (e.g., *MOT:, *CHI:) - should already be stripped but just in case
    text = re.sub(r'^\*[A-Z]+:\s*', '', text)
    
    # Remove xxx, yyy, www (unidentifiable material)
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'\byyyy\b', '', text)
    text = re.sub(r'\bwww\b', '', text)
    
    # Handle error codes with omitted words: [* 0word] or [* &=0word] -> insert "word"
    # Example: "I want [* 0to] go" -> "I want to go"
    text = re.sub(r'\[\*[^\]]*?&?=?0(\w+)[^\]]*\]', r'\1', text)
    
    # Handle omitted words in main text with &= prefix: &=0word -> word
    # Example: "I go &=0for a walk" -> "I go for a walk"
    text = re.sub(r'&=0(\w+)', r'\1', text)
    
    # Handle word repetition: word [/] word -> word
    # The [/] marker indicates the previous word is repeated, keep only one instance
    text = re.sub(r'(\w+)\s*\[/\]\s*\1\b', r'\1', text)
    
    # Handle phrase repetition: <phrase> [/] phrase -> phrase
    # Remove the bracketed repeated material and the [/] marker
    text = re.sub(r'<[^>]+>\s*\[/\]\s*', '', text)
    
    # Remove omitted words in main text (0word without brackets or &= prefix)
    text = re.sub(r'\b0\w+', '', text)
    
    # Remove simple events (&=action) - must come AFTER handling &=0word
    text = re.sub(r'&=[^\s]+', '', text)
    
    # Remove filled pauses (&-filler)
    text = re.sub(r'&-\w+', '', text)
    
    # Remove phonological fragments (&+fragment)
    text = re.sub(r'&\+[^\s]+', '', text)
    
    # Remove other non-word strings (&word)
    text = re.sub(r'&\w+', '', text)
    
    # Remove remaining bracket annotations [text], [: text], [:: text], [=! text], etc.
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Remove remaining angle brackets (repeated/revised material markers)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove special utterance terminators and symbols
    text = re.sub(r'\+\.\.\.?', '', text)  # trailing off
    text = re.sub(r'\+/\.?', '', text)     # interruption
    text = re.sub(r'\+//\.?', '', text)    # self-interruption
    text = re.sub(r'\+\^', '', text)       # quick uptake
    text = re.sub(r'\+,', '', text)        # self completion
    text = re.sub(r'\+\+', '', text)       # other completion
    text = re.sub(r'\+"', '', text)        # quotation markers
    text = re.sub(r'"\+', '', text)
    
    # Remove special symbols for stress, tone, etc.
    text = re.sub(r'[↓↑ˈˌ]', '', text)
    
    # Remove pause markers
    text = re.sub(r'\(\.+\)', '', text)           # (.) (..) (...)
    text = re.sub(r'\(\d+\.?\d*\)', '', text)     # (2.4) duration
    
    # Remove prolongation markers (:)
    text = re.sub(r'(\w):+', r'\1', text)
    
    # Remove broken word markers (^)
    text = re.sub(r'\^', '', text)
    
    # Remove block markers (≠)
    text = re.sub(r'≠', '', text)
    
    # Remove repeated segment markers (↫)
    text = re.sub(r'↫', '', text)
    
    # Handle compounds: word+word -> word (or keep as "wordword")
    # For now, replace + with space to keep both parts
    text = re.sub(r'(\w)\+(\w)', r'\1 \2', text)
    
    # Remove incomplete word markers (word(text))
    # Keep the full word with intended meaning
    text = re.sub(r'(\w+)\([^\)]+\)', r'\1', text)
    
    # Remove shortening markers ((text)word)
    text = re.sub(r'\([^\)]+\)(\w+)', r'\1', text)
    
    # Remove special form markers (@c, @l, etc. attached to words)
    text = re.sub(r'@[a-z]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text.strip()


def load_manchester_corpus(corpus_path: str = DEFAULT_CORPUS_PATH, cache_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load all transcript files from Manchester corpus, preserving temporal ordering.
    
    CRITICAL: Loads original transcript files (.cha) to maintain the chronological 
    sequence of child and mother utterances. Each utterance gets a line_num that preserves
    the original ordering from the transcript file.
    
    The cleaned corpus is cached to avoid reprocessing. If cache_dir is provided and
    a cached corpus exists, it will be loaded instead of reprocessing the raw files.
    
    Args:
        corpus_path: Path to CHILDES Manchester corpus directory
        cache_dir: Directory to save/load cached cleaned corpus (if None, no caching)
        
    Returns:
        dict: {child_name: combined_df}
        Each combined_df has columns: ['filename', 'speaker', 'sentence', 'line_num', 'speaker_type']
        where speaker_type is either 'child' or 'mother', and line_num preserves temporal order
    """
    # Check for cached cleaned corpus
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'cleaned_corpus.csv')
        if os.path.exists(cache_file):
            print(f"Loading cached cleaned corpus from {cache_file}...")
            corpus_df = pd.read_csv(cache_file)
            
            # Reconstruct the dict structure: {child_name: child_df}
            corpus_data = {}
            for child_name in corpus_df['child_name'].unique():
                child_df = corpus_df[corpus_df['child_name'] == child_name].copy()
                # Drop child_name column since it's now the dict key
                child_df = child_df.drop(columns=['child_name'])
                corpus_data[child_name] = child_df
            
            print(f"Loaded {len(corpus_data)} children from cache")
            return corpus_data
    
    print("Processing raw corpus files (this will be cached for future use)...")
    corpus_data = {}
    children = [f.name for f in os.scandir(corpus_path) if f.is_dir()]
    
    # Row identifiers from CHAT format
    MOTHER_ROW = '*MOT:'
    CHILD_ROW = '*CHI:'
    
    for child in children:
        child_path = os.path.join(corpus_path, child)
        combined_rows = []
        
        # Read all transcript files for this child
        for trans_file in os.scandir(child_path):
            if not trans_file.name.endswith('.cha'):
                continue
                
            try:
                # Read transcript file
                transcript = pd.read_csv(
                    trans_file.path,
                    sep="\t",
                    on_bad_lines='warn',
                    names=["speaker", "sentence"],
                ).dropna()
                
                # Add filename and line number to preserve ordering
                transcript["filename"] = trans_file.path
                transcript["line_num"] = range(len(transcript))
                
                # Extract child and mother utterances WITH their line numbers
                child_utts = transcript[transcript.speaker.str.contains(CHILD_ROW, regex=False)].copy()
                child_utts["speaker_type"] = "child"
                
                mot_utts = transcript[transcript.speaker.str.contains(MOTHER_ROW, regex=False)].copy()
                mot_utts["speaker_type"] = "mother"
                
                # Combine and keep chronological order
                file_utts = pd.concat([child_utts, mot_utts], ignore_index=False)
                file_utts = file_utts.sort_values('line_num')  # Maintain temporal order
                
                combined_rows.append(file_utts)
                
            except Exception as e:
                print(f"Warning: Could not read {trans_file.path}: {e}")
                continue
        
        if combined_rows:
            # Combine all transcript files for this child
            combined_df = pd.concat(combined_rows, ignore_index=True)
            
            # Clean CHILDES annotations from sentences
            print(f"  Cleaning CHILDES annotations for {child}...")
            combined_df['sentence'] = combined_df['sentence'].apply(clean_childes_annotations)
            
            # Remove empty sentences after cleaning
            combined_df = combined_df[combined_df['sentence'].str.strip() != ''].copy()
            
            # Keep only needed columns
            combined_df = combined_df[['filename', 'speaker', 'sentence', 'line_num', 'speaker_type']]
            
            corpus_data[child] = combined_df
            
            n_child = len(combined_df[combined_df.speaker_type == 'child'])
            n_mother = len(combined_df[combined_df.speaker_type == 'mother'])
            print(f"Loaded {child}: {n_child} child utterances, {n_mother} mother utterances "
                  f"({len(combined_df)} total in temporal order)")
    
    # Save cleaned corpus to cache
    if cache_dir and corpus_data:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'cleaned_corpus.csv')
        print(f"\nSaving cleaned corpus to cache: {cache_file}")
        
        # Combine all children into one DataFrame with child_name column
        all_children = []
        for child_name, child_df in corpus_data.items():
            child_df_copy = child_df.copy()
            child_df_copy['child_name'] = child_name
            all_children.append(child_df_copy)
        
        combined_corpus = pd.concat(all_children, ignore_index=True)
        # Reorder columns to put child_name first
        cols = ['child_name'] + [col for col in combined_corpus.columns if col != 'child_name']
        combined_corpus = combined_corpus[cols]
        
        combined_corpus.to_csv(cache_file, index=False)
        print(f"Cached {len(corpus_data)} children ({len(combined_corpus)} total utterances) for future use")
    
    return corpus_data


def create_masked_input(sentence, det_char_start, det_char_end, model_type, mask_token="<mask>"):
    """
    Create masked version of sentence for model input.
    
    Args:
        sentence: Original sentence string
        det_char_start: Character index where determiner starts
        det_char_end: Character index where determiner ends
        model_type: 'mlm', 'ar', or 'seq2seq'
        mask_token: Token to use for masking (for MLM models)
        
    Returns:
        str: Masked sentence
    """
    AR_BLANK_TOKEN = '___'  # Underscore blank for fill-in-the-blank prompting
    
    if model_type == 'mlm':
        # Replace determiner with mask token
        masked = sentence[:det_char_start] + mask_token + sentence[det_char_end:]
    elif model_type in {'ar', 'seq2seq'}:
        # Replace determiner with ___ blank
        masked = sentence[:det_char_start] + AR_BLANK_TOKEN + sentence[det_char_end:]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: mlm, ar, seq2seq")
    
    return masked


def get_determiner_token_ids(tokenizer):
    """
    Get token IDs for determiners {a, an, the} in the tokenizer's vocabulary.
    
    Returns:
        dict: {det_string: token_id}
    """
    # Try various tokenization strategies
    det_tokens = {}
    
    for det in ['a', 'an', 'the']:
        # Try with space prefix (common in RoBERTa, GPT-2)
        variants = [det, ' ' + det, 'Ġ' + det, det.capitalize(), ' ' + det.capitalize()]
        
        for variant in variants:
            try:
                token_id = tokenizer.convert_tokens_to_ids(variant)
                if token_id != tokenizer.unk_token_id:
                    det_tokens[det] = token_id
                    break
            except:
                continue
    
    # Verify we found all three
    if len(det_tokens) != 3:
        print(f"Warning: Could only find {len(det_tokens)}/3 determiner tokens")
        print(f"Found: {det_tokens}")
    
    return det_tokens


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_config, device=None, revision=None):
    """Load a model + tokenizer for CAC scoring (delegates to the carved `cac` package)."""
    return cac.load_model(model_config, device=device, revision=revision)


# =============================================================================
# DETERMINER PROBABILITY EXTRACTION
# =============================================================================

def extract_determiner_probabilities(model, tokenizer, model_type, masked_sentence, det_token_ids=None):
    """Determiner CAC scoring via the carved `cac` core (a/an merged into the "a" category)."""
    return cac.score_candidates(model, tokenizer, model_type, masked_sentence,
                                DETERMINER_CANDIDATES, device=device)


def choose_determiner_predictions(probs: Dict[str, float]) -> Tuple[str, str]:
    """Return (argmax_choice, sampled_choice) from {'a': p, 'the': p} probabilities."""
    p_a = float(probs.get('a', 0.5))
    p_the = float(probs.get('the', 0.5))

    det_argmax = 'a' if p_a > p_the else 'the'

    p_arr = np.array([p_a, p_the], dtype=float)
    total = float(p_arr.sum())
    if total <= 0 or not np.isfinite(total):
        p_arr = np.array([0.5, 0.5], dtype=float)
    else:
        p_arr = p_arr / total

    det_sample = np.random.choice(['a', 'the'], p=p_arr)
    return det_argmax, det_sample


# =============================================================================
# DISCOURSE CONTEXT BUILDING
# =============================================================================


def _estimate_max_context_lines(model_configs_path, corpus_sample):
    """Delegates to cac.estimate_max_context_lines (carved into the cac package)."""
    return cac.estimate_max_context_lines(model_configs_path, corpus_sample)


def prepare_discourse_inputs_all_sessions(corpus: Dict[str, pd.DataFrame],
                                          det_noun_locations: pd.DataFrame,
                                          discourse_output_dir: str,
                                          model_configs_path: str = 'model_configs.json',
                                          discourse_label_style: str = 'spoken') -> pd.DataFrame:
    """
    Pre-compute discourse inputs for all sessions and cache in one file.

    The cache stores **only** the integer line-number boundary of the context
    window (``context_start_line_num``) instead of serialising the full text.
    This keeps the cache file small (a few MB instead of ~1 GB).  At inference
    time the caller reconstructs the actual utterances from the corpus using the
    ``(child_name, filename, context_start_line_num, line_num)`` quad.

    Discourse mode: for each det-noun pair, include all prior utterances in the
    same recording session (regardless of speaker). This cached representation is
    model-agnostic and can be truncated per model at inference time.

    Args:
        corpus: Dict mapping child_name -> full session utterances DataFrame
        det_noun_locations: All det-noun locations to score
        discourse_output_dir: Output directory used for caching
        model_configs_path: Path to model_configs.json (used to estimate line cap)
        discourse_label_style: Preserved for API symmetry with label-style runs.

    Returns:
        DataFrame with one row per det-noun prediction target.
        Key columns: line_num (target), context_start_line_num (-1 = no context).
        At inference time, reconstruct context by slicing the corpus between
        context_start_line_num (inclusive) and line_num (exclusive).
    """
    cache_file = os.path.join(discourse_output_dir, 'discourse_cached_inputs.csv')

    if os.path.exists(cache_file):
        print(f"  Loading cached discourse inputs from {cache_file}")
        cached_df = pd.read_csv(cache_file)
        required_cols = {
            'sentence_id', 'child_name', 'filename', 'line_num', 'speaker', 'noun',
            'original_sentence', 'original_det', 'masked_position',
            'det_char_start', 'det_char_end', 'context_start_line_num'
        }
        if required_cols.issubset(set(cached_df.columns)):
            return cached_df

        print("  Cached discourse inputs are outdated; rebuilding cache with new schema...")

    print("  Computing discourse cached inputs for all sessions...")

    # Dynamically estimate how many prior lines any model can ever attend to.
    # Build a flat sample from the corpus for the avg-tokens heuristic.
    sample_df = pd.concat(list(corpus.values()), ignore_index=True) if corpus else pd.DataFrame(columns=['sentence'])
    max_context_lines = cac.estimate_max_context_lines(model_configs_path, sample_df)
    targets = det_noun_locations.reset_index(drop=True)
    context_start = cac.compute_context_starts(corpus, targets, max_window_lines=max_context_lines)

    # Build output rows in original target order.
    prepared_rows = []
    for i, row in targets.iterrows():
        prepared_rows.append({
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
        })

    if prepared_rows:
        prepared_df = pd.DataFrame(prepared_rows)
        prepared_df.to_csv(cache_file, index=False)
        print(f"  Saved {len(prepared_df)} discourse cached inputs to {cache_file}")
        return prepared_df

    return pd.DataFrame()


def format_discourse_input(context_utterances, target_speaker, target_sentence, model_type,
                          tokenizer=None, max_tokens=None, discourse_label_style='spoken'):
    """Determiner discourse formatter — delegates to cac.format_context_input."""
    if discourse_label_style not in {'spoken', 'childes'}:
        raise ValueError(
            f"Unsupported discourse_label_style '{discourse_label_style}'. Expected 'spoken' or 'childes'.")
    if discourse_label_style == 'childes':
        _resolver = lambda s: '*CHI' if str(s).lower() == 'child' else '*MOT'
    else:
        _resolver = lambda s: 'CHILD' if str(s).lower() == 'child' else 'MOTHER'
    return cac.format_context_input(context_utterances, target_speaker, target_sentence, model_type,
                                    speaker_label=_resolver, tokenizer=tokenizer, max_tokens=max_tokens)


# =============================================================================
# SAVING AND LOADING RESULTS
# =============================================================================

def save_predictions(predictions: List[Dict], output_path: str, model_config: Dict,
                    child_name: str, speaker: str):
    """Save predictions to file."""
    base_name = model_config['name'].split('/')[-1]
    model_type = model_config['type']
    
    # Create output directory matching notebook structure
    output_dir = os.path.join(output_path, model_type, base_name, child_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions CSV (no date stamp to match notebook)
    pred_file = os.path.join(output_dir, f"{speaker}_predictions.csv")
    df = pd.DataFrame(predictions)
    df.to_csv(pred_file, index=False)
    print(f"  Saved predictions to {pred_file}")
    
    # Extract pairs for deterministic method (use original column names)
    det_pairs = [f"{row['predicted_det_argmax']} {row['noun']}" 
                 for row in predictions]
    det_pairs_file = os.path.join(output_dir, f"{speaker}_deterministic_pairs.txt")
    with open(det_pairs_file, 'w') as f:
        f.write('\n'.join(det_pairs))
    print(f"  Saved {len(det_pairs)} deterministic pairs to {det_pairs_file}")
    
    # Extract pairs for probabilistic method (use original column names)
    prob_pairs = [f"{row['predicted_det_sample']} {row['noun']}" 
                  for row in predictions]
    prob_pairs_file = os.path.join(output_dir, f"{speaker}_probabilistic_pairs.txt")
    with open(prob_pairs_file, 'w') as f:
        f.write('\n'.join(prob_pairs))
    print(f"  Saved {len(prob_pairs)} probabilistic pairs to {prob_pairs_file}")


def _load_model_name_mapping(config_file: str = 'model_configs.json') -> Dict[str, str]:
    """Map base model names to full Hugging Face paths from model configs."""
    name_mapping = {}
    if not os.path.exists(config_file):
        return name_mapping

    try:
        with open(config_file, 'r') as f:
            model_configs = json.load(f)
        for config in model_configs:
            full_name = config.get('name', '')
            if not full_name:
                continue
            base_name = full_name.split('/')[-1] if '/' in full_name else full_name
            name_mapping[base_name] = full_name
    except Exception:
        pass

    return name_mapping


def _iter_model_dirs(base_path: str):
    """Yield (model_type, model_name, model_dir) triples for experiment outputs."""
    if not os.path.exists(base_path):
        return

    # Only iterate canonical model-family folders used by current experiments.
    allowed_model_types = {'mlm', 'ar', 'seq2seq'}
    for model_type in sorted(os.listdir(base_path)):
        if model_type not in allowed_model_types:
            continue

        type_dir = os.path.join(base_path, model_type)
        if not os.path.isdir(type_dir):
            continue

        for model_name in os.listdir(type_dir):
            model_dir = os.path.join(type_dir, model_name)
            if os.path.isdir(model_dir):
                yield model_type, model_name, model_dir


def _normalize_det(value) -> str:
    token = str(value).strip().lower()
    return 'a' if token in {'a', 'an'} else token


def load_overlap_results(base_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load overlap-experiment accuracy and overlap summaries (isolated/discourse).

    Args:
        base_path: './output/manchester_masked' or './output/manchester_discourse'

    Returns:
        (accuracy_df, overlap_df) or (None, None) if no results exist.
    """
    accuracy_data = []
    overlap_data = []

    if not os.path.exists(base_path):
        return None, None

    name_mapping = _load_model_name_mapping()

    for model_type, model_name, model_dir in _iter_model_dirs(base_path):
        full_model_name = name_mapping.get(model_name, model_name)

        # Build speaker-level accuracy rows from per-child prediction files.
        for child_name in os.listdir(model_dir):
            child_dir = os.path.join(model_dir, child_name)
            if not os.path.isdir(child_dir):
                continue

            for speaker, pred_name in {'child': 'child_predictions.csv', 'mother': 'mother_predictions.csv'}.items():
                pred_file = os.path.join(child_dir, pred_name)
                if not os.path.exists(pred_file):
                    continue

                try:
                    pred_df = pd.read_csv(pred_file)
                except Exception:
                    continue

                if pred_df.empty or 'original_det' not in pred_df.columns:
                    continue

                orig = pred_df['original_det'].map(_normalize_det)
                acc_argmax = np.nan
                acc_sample = np.nan

                if 'predicted_det_argmax' in pred_df.columns:
                    acc_argmax = float((orig == pred_df['predicted_det_argmax'].map(_normalize_det)).mean())
                if 'predicted_det_sample' in pred_df.columns:
                    acc_sample = float((orig == pred_df['predicted_det_sample'].map(_normalize_det)).mean())

                accuracy_data.append(pd.DataFrame([{
                    'model_name': full_model_name,
                    'model_type': model_type,
                    'child_name': child_name,
                    'speaker': speaker,
                    'n_predictions': int(len(pred_df)),
                    'accuracy_argmax': acc_argmax,
                    'accuracy_sample': acc_sample,
                }]))

        overlap_file = os.path.join(model_dir, 'overlap_summary.csv')
        if os.path.exists(overlap_file):
            df = pd.read_csv(overlap_file)
            if 'model_name' in df.columns and model_name in name_mapping:
                df['model_name'] = full_model_name
            overlap_data.append(df)

    accuracy_df = pd.concat(accuracy_data, ignore_index=True) if accuracy_data else None
    overlap_df = pd.concat(overlap_data, ignore_index=True) if overlap_data else None
    return accuracy_df, overlap_df


def load_tpr_results(tpr_base_path: str = './output/manchester_tpr') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load TPR long-form results plus per-model determiner accuracy derived from TPR predictions.

    Args:
        tpr_base_path: Directory containing tpr_all_results.csv and tpr_predictions_*.csv files.

    Returns:
        (tpr_all_df, tpr_accuracy_df)
        - tpr_all_df: method-separated TPR rows when possible (or None)
        - tpr_accuracy_df: rows with columns [model_name, speaker, accuracy_argmax, accuracy_sample, n_predictions]
    """
    if not os.path.exists(tpr_base_path):
        return None, None

    name_mapping = _load_model_name_mapping()

    tpr_file = os.path.join(tpr_base_path, 'tpr_all_results.csv')
    tpr_all_df = pd.read_csv(tpr_file) if os.path.exists(tpr_file) else None

    if tpr_all_df is not None and not tpr_all_df.empty:
        if 'model_name' in tpr_all_df.columns:
            tpr_all_df['model_name'] = tpr_all_df['model_name'].astype(str).map(
                lambda m: name_mapping.get(m.split('/')[-1], m)
            )

        if 'speaker' not in tpr_all_df.columns and 'child_or_mother' in tpr_all_df.columns:
            tpr_all_df['speaker'] = tpr_all_df['child_or_mother']

        method_in_aggregate = 'method' in tpr_all_df.columns
        if not method_in_aggregate:
            # Preserve deterministic aggregate rows (including Human) but never mix with probabilistic rows.
            tpr_all_df['method'] = 'det'
        else:
            tpr_all_df['method'] = tpr_all_df['method'].map({
                'argmax': 'det',
                'sample': 'prob',
                'deterministic': 'det',
                'probabilistic': 'prob'
            }).fillna(tpr_all_df['method'])

        if 'tpr' not in tpr_all_df.columns and {'n_transitions_to_the', 'n_transitions_to_a', 'n_total'}.issubset(tpr_all_df.columns):
            tpr_all_df['tpr'] = (
                tpr_all_df['n_transitions_to_the'] + tpr_all_df['n_transitions_to_a']
            ) / tpr_all_df['n_total'].replace(0, np.nan)

    pred_rows = []
    tpr_prob_rows = []
    for filename in os.listdir(tpr_base_path):
        if not (filename.startswith('tpr_predictions_') and filename.endswith('.csv')):
            continue

        pred_path = os.path.join(tpr_base_path, filename)
        model_base_name = filename[len('tpr_predictions_'):-4]
        full_model_name = name_mapping.get(model_base_name, model_base_name)

        try:
            preds = pd.read_csv(pred_path)
        except Exception:
            continue

        if preds.empty or 'original_det' not in preds.columns:
            continue

        if 'speaker' in preds.columns:
            speakers = [s for s in preds['speaker'].dropna().unique()]
        else:
            speakers = ['child']
            preds['speaker'] = 'child'

        for speaker in speakers:
            sub = preds[preds['speaker'] == speaker]
            if sub.empty:
                continue

            orig = sub['original_det'].map(_normalize_det)
            acc_argmax = np.nan
            acc_sample = np.nan

            if 'predicted_det_argmax' in sub.columns:
                acc_argmax = float((orig == sub['predicted_det_argmax'].map(_normalize_det)).mean())

            if 'predicted_det_sample' in sub.columns:
                acc_sample = float((orig == sub['predicted_det_sample'].map(_normalize_det)).mean())

            pred_rows.append({
                'model_name': full_model_name,
                'speaker': str(speaker).strip().lower(),
                'accuracy_argmax': acc_argmax,
                'accuracy_sample': acc_sample,
                'n_predictions': int(len(sub))
            })

            # If the aggregate TPR file has no method column, recover probabilistic TPR from predictions
            # so deterministic and probabilistic analyses stay strictly separated.
            if tpr_all_df is not None and not tpr_all_df.empty and not method_in_aggregate and 'previous_det' in sub.columns:
                for method_label, pred_col in [('det', 'predicted_det_argmax'), ('prob', 'predicted_det_sample')]:
                    if pred_col not in sub.columns:
                        continue

                    work = sub[['child_name', 'model_type', 'previous_det', pred_col]].copy()
                    work = work.rename(columns={pred_col: 'predicted_det'})
                    work['previous_det'] = work['previous_det'].map(_normalize_det)
                    work['predicted_det'] = work['predicted_det'].map(_normalize_det)
                    work = work[
                        work['previous_det'].isin(['a', 'the'])
                        & work['predicted_det'].isin(['a', 'the'])
                    ]

                    if work.empty:
                        continue

                    for child_name, g in work.groupby('child_name', dropna=False):
                        n_previous_a = int((g['previous_det'] == 'a').sum())
                        n_previous_the = int((g['previous_det'] == 'the').sum())
                        n_total = int(len(g))

                        n_transitions_to_the = int(((g['previous_det'] == 'a') & (g['predicted_det'] == 'the')).sum())
                        n_transitions_to_a = int(((g['previous_det'] == 'the') & (g['predicted_det'] == 'a')).sum())
                        n_maintained = int((g['previous_det'] == g['predicted_det']).sum())

                        tpr_prob_rows.append({
                            'child_name': child_name,
                            'filename': 'aggregated_from_predictions',
                            'speaker': str(speaker).strip().lower(),
                            'source': 'predictions',
                            'model_name': full_model_name,
                            'model_type': g['model_type'].iloc[0] if 'model_type' in g.columns and not g.empty else 'unknown',
                            'n_attested_nouns': np.nan,
                            'tpr_the': (n_transitions_to_the / n_previous_a) if n_previous_a else np.nan,
                            'tpr_a': (n_transitions_to_a / n_previous_the) if n_previous_the else np.nan,
                            'tpr_overall': (n_maintained / n_total) if n_total else np.nan,
                            'tpr': (n_maintained / n_total) if n_total else np.nan,
                            'n_transitions_to_the': n_transitions_to_the,
                            'n_transitions_to_a': n_transitions_to_a,
                            'n_previous_a': n_previous_a,
                            'n_previous_the': n_previous_the,
                            'n_maintained': n_maintained,
                            'n_total': n_total,
                            'method': method_label,
                        })

    tpr_accuracy_df = pd.DataFrame(pred_rows) if pred_rows else None

    if tpr_prob_rows:
        tpr_prob_df = pd.DataFrame(tpr_prob_rows)
        # When aggregate lacks explicit method, keep Human deterministic rows from aggregate and
        # append model-specific det/prob rows recovered from predictions.
        if tpr_all_df is None or tpr_all_df.empty:
            tpr_all_df = tpr_prob_df
        elif 'method' in tpr_all_df.columns and not method_in_aggregate:
            human_mask = tpr_all_df['model_name'].astype(str).str.lower() == 'human'
            tpr_all_df = pd.concat([tpr_all_df[human_mask].copy(), tpr_prob_df], ignore_index=True, sort=False)

    if tpr_all_df is not None and not tpr_all_df.empty:
        if 'tpr' not in tpr_all_df.columns:
            tpr_all_df['tpr'] = np.nan
        if {'n_transitions_to_the', 'n_transitions_to_a', 'n_total'}.issubset(tpr_all_df.columns):
            tpr_all_df['tpr'] = tpr_all_df['tpr'].fillna(
                (tpr_all_df['n_transitions_to_the'] + tpr_all_df['n_transitions_to_a'])
                / tpr_all_df['n_total'].replace(0, np.nan)
            )
        if 'tpr_overall' in tpr_all_df.columns:
            tpr_all_df['tpr'] = tpr_all_df['tpr'].fillna(tpr_all_df['tpr_overall'])

    return tpr_all_df, tpr_accuracy_df



# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def generate_model_summaries(output_path: str, model_config: Dict):
    """
    Write deterministic pairs files used downstream by the inference pipeline.

    Accuracy, overlap, and TPR are all computed analytically by
    compute_analytical_metrics.py and no longer derived here.
    """
    full_model_name = model_config['name']
    model_name = full_model_name.split('/')[-1]
    model_type = model_config['type']
    model_dir = os.path.join(output_path, model_type, model_name)

    if not os.path.exists(model_dir):
        return

    # Write deterministic and probabilistic pairs files for each child/speaker
    # so that any legacy callers continue to find them.
    for child_name in os.listdir(model_dir):
        child_dir = os.path.join(model_dir, child_name)
        if not os.path.isdir(child_dir):
            continue
        for speaker in ['child', 'mother']:
            pred_file = os.path.join(child_dir, f"{speaker}_predictions.csv")
            if not os.path.exists(pred_file):
                continue
            try:
                df = pd.read_csv(pred_file)
                for method, col in [('deterministic', 'predicted_det_argmax'),
                                    ('probabilistic', 'predicted_det_sample')]:
                    if col not in df.columns or 'noun' not in df.columns:
                        continue
                    pairs = df[[col, 'noun']].dropna()
                    pairs_file = os.path.join(child_dir, f"{speaker}_{method}_pairs.txt")
                    with open(pairs_file, 'w') as f:
                        for _, row in pairs.iterrows():
                            f.write(f"{row[col]} {row['noun']}\n")
            except Exception as e:
                print(f"    Warning: could not write pairs for {child_dir}: {e}")


# =============================================================================
# TRANSITIONAL PROBABILITY OF REFERENCE (TPR) ANALYSIS
# =============================================================================

def find_attested_nouns_per_session(corpus: Dict[str, pd.DataFrame], 
                                     det_noun_locations: pd.DataFrame) -> Dict[str, Dict[str, set]]:
    """
    Find nouns that BOTH the mother AND the child used with both determiners ('a/an' and 'the') 
    in each recording session, with CUMULATIVE filtering across chronologically-ordered transcripts.
    
    For each transcript, the subset of attested nouns includes:
    - Nouns attested in the current transcript (both speakers used both determiners)
    - Nouns attested in ANY previous transcript (already confirmed usable with both determiners)
    
    This ensures that once a noun is attested, it remains in the valid set for all subsequent transcripts.
    
    Args:
        corpus: Dict mapping child_name -> DataFrame
        det_noun_locations: DataFrame with extracted det-noun pairs
        
    Returns:
        Dict[child_name][filename] -> Set[nouns] that both mother and child have used with both determiners
                                      (in current or any previous transcript)
    """
    attested_nouns = {}
    
    for child_name in corpus.keys():
        attested_nouns[child_name] = {}
        
        # Get all locations for this child (both child and mother)
        dyad_locs = det_noun_locations[
            (det_noun_locations['child_name'] == child_name)
        ]
        
        # Get unique filenames and sort chronologically (filenames are YYMMDD.cha format)
        filenames = sorted(dyad_locs['filename'].unique())
        
        # Track cumulative attested nouns across all previous + current transcripts
        cumulative_attested = set()
        
        # Process each filename in chronological order
        for filename in filenames:
            session_locs = dyad_locs[dyad_locs['filename'] == filename]
            
            # Track determiners used by each speaker for each noun IN THIS SESSION
            child_noun_dets = {}
            mother_noun_dets = {}
            
            for _, row in session_locs.iterrows():
                noun = row['noun']
                det = row['det']
                speaker = row['speaker_type']
                
                # Skip if speaker_type is not child or mother
                if pd.isna(speaker) or speaker not in ['child', 'mother']:
                    continue
                
                # Normalize: 'a'/'an' -> 'a_type', 'the' -> 'the_type'
                det_type = 'the_type' if det == 'the' else 'a_type'
                
                # Track by speaker
                if speaker == 'child':
                    if noun not in child_noun_dets:
                        child_noun_dets[noun] = set()
                    child_noun_dets[noun].add(det_type)
                elif speaker == 'mother':
                    if noun not in mother_noun_dets:
                        mother_noun_dets[noun] = set()
                    mother_noun_dets[noun].add(det_type)
            
            # Find nouns newly attested in THIS session
            # (where BOTH child AND mother used BOTH determiners)
            child_attested = {noun for noun, dets in child_noun_dets.items() 
                             if 'a_type' in dets and 'the_type' in dets}
            mother_attested = {noun for noun, dets in mother_noun_dets.items() 
                              if 'a_type' in dets and 'the_type' in dets}
            
            # Newly attested = intersection of child and mother in this session
            newly_attested = child_attested & mother_attested
            
            # Add newly attested nouns to cumulative set
            cumulative_attested.update(newly_attested)
            
            # Store cumulative set for this session (includes all previous + current)
            attested_nouns[child_name][filename] = cumulative_attested.copy()
    
    return attested_nouns


def calculate_tpr_per_session(session_locations: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate TPR for both speakers in a single recording session.
    
    Tracks last mention by OTHER speaker for each noun, counts transitions.
    
    Args:
        session_locations: Det-noun locations filtered to one session with attested nouns
        
    Returns:
        Dict with keys 'child' and 'mother', each containing TPR statistics
    """
    # Track last mention by other speaker for each noun
    # {noun: {'det': 'a'|'the', 'speaker': 'child'|'mother', 'line_num': int, 'sentence': str}}
    last_mention_by_other = {}
    
    # Results for each speaker
    results = {
        'child': {
            'transitions_to_the': 0,
            'transitions_to_a': 0,
            'previous_a': 0,
            'previous_the': 0,
            'n_maintained': 0
        },
        'mother': {
            'transitions_to_the': 0,
            'transitions_to_a': 0,
            'previous_a': 0,
            'previous_the': 0,
            'n_maintained': 0
        },
        'other_adult': {
            'transitions_to_the': 0,
            'transitions_to_a': 0,
            'previous_a': 0,
            'previous_the': 0,
            'n_maintained': 0
        }
    }
    
    # Sort chronologically
    session_locations = session_locations.sort_values('line_num')
    
    # Process each mention
    for idx, row in session_locations.iterrows():
        speaker = row['speaker_type']
        if pd.isna(speaker) or speaker not in ['child', 'mother', 'other_adult']:
            continue
            
        noun = row['noun']
        det = row['det']
        det_type = 'the' if det == 'the' else 'a'  # Normalize
        
        target_prior_speakers = []
        if speaker == 'child':
            target_prior_speakers = ['mother']
        elif speaker == 'mother':
            target_prior_speakers = ['child']
        elif speaker == 'other_adult':
            target_prior_speakers = ['mother']
        
        # Check if matching OTHER speaker mentioned this noun before
        if noun in last_mention_by_other:
            prev_mention = last_mention_by_other[noun]
            if prev_mention['speaker'] in target_prior_speakers:
                # Valid TPR observation
                prev_det = prev_mention['det']
                
                if prev_det == 'a':
                    results[speaker]['previous_a'] += 1
                    if det_type == 'the':
                        results[speaker]['transitions_to_the'] += 1
                    else:
                        results[speaker]['n_maintained'] += 1
                else:  # prev_det == 'the'
                    results[speaker]['previous_the'] += 1
                    if det_type == 'a':
                        results[speaker]['transitions_to_a'] += 1
                    else:
                        results[speaker]['n_maintained'] += 1
        
        # Update last mention for this noun
        last_mention_by_other[noun] = {
            'det': det_type,
            'speaker': speaker,
            'line_num': row['line_num'],
            'sentence': row['sentence']
        }
    
    # Calculate TPR for each speaker
    output = {}
    for speaker in ['child', 'mother', 'other_adult']:
        r = results[speaker]
        tpr_the = r['transitions_to_the'] / r['previous_a'] if r['previous_a'] > 0 else None
        tpr_a = r['transitions_to_a'] / r['previous_the'] if r['previous_the'] > 0 else None
        n_total = r['previous_a'] + r['previous_the']
        tpr_overall = (r['transitions_to_the'] + r['transitions_to_a']) / n_total if n_total > 0 else None
        
        output[speaker] = {
            'tpr_the': tpr_the,
            'tpr_a': tpr_a,
            'tpr_overall': tpr_overall,
            'n_transitions_to_the': r['transitions_to_the'],
            'n_transitions_to_a': r['transitions_to_a'],
            'n_previous_a': r['previous_a'],
            'n_previous_the': r['previous_the'],
            'n_maintained': r['n_maintained'],
            'n_total': n_total
        }
    
    return output


def calculate_tpr_human_baseline(corpus: Dict[str, pd.DataFrame],
                                 det_noun_locations: pd.DataFrame,
                                 attested_nouns: Dict[str, Dict[str, set]],
                                 tpr_output_dir: str,
                                 results_tpr_dir: str = None) -> pd.DataFrame:
    """
    Calculate TPR from original human transcripts per session.

    Saves results to results_tpr_dir/tpr_human_baseline.csv (committed results).
    Falls back to tpr_output_dir if results_tpr_dir is not provided.

    Args:
        corpus: Dict mapping child_name -> DataFrame
        det_noun_locations: DataFrame with extracted det-noun pairs
        attested_nouns: Dict[child_name][filename] -> Set[attested nouns]
        tpr_output_dir: Pipeline cache directory
        results_tpr_dir: Committed results directory (writes tpr_human_baseline.csv here)

    Returns:
        DataFrame with per-session TPR results for all children
    """
    all_results = []
    
    for child_name, child_df in corpus.items():
        print(f"  Calculating TPR for {child_name}...")
        
        # Get locations for this child
        child_locs = det_noun_locations[det_noun_locations['child_name'] == child_name]
        
        # Process each session
        for filename in child_locs['filename'].unique():
            # Get attested nouns for this session
            if filename not in attested_nouns[child_name]:
                continue
            attested = attested_nouns[child_name][filename]
            
            if len(attested) == 0:
                continue  # No nouns with both determiners in this session
            
            # Filter to this session AND attested nouns only
            session_locs = child_locs[
                (child_locs['filename'] == filename) &
                (child_locs['noun'].isin(attested))
            ]
            
            if len(session_locs) == 0:
                continue
            
            # Calculate TPR for this session
            session_tpr = calculate_tpr_per_session(session_locs)
            
            # Add results for all speakers
            for speaker in ['child', 'mother', 'other_adult']:
                result = {
                    'child_name': child_name,
                    'filename': os.path.basename(filename),
                    'speaker': speaker,
                    'source': 'human',
                    'model_name': 'Human',
                    'model_type': 'baseline',
                    'n_attested_nouns': len(attested),
                    **session_tpr[speaker]
                }
                all_results.append(result)
    
    # Save all human results in one file
    results_df = pd.DataFrame(all_results)
    out_dir = results_tpr_dir if results_tpr_dir is not None else tpr_output_dir
    os.makedirs(out_dir, exist_ok=True)
    human_file = os.path.join(out_dir, 'tpr_human_baseline.csv')
    results_df.to_csv(human_file, index=False)
    print(f"  Saved human baseline to {human_file}")
    
    return results_df


def aggregate_tpr_results(per_session_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-session TPR results per child.
    
    Produces table matching format:
    Dyad | N | TPR (M→C) | TPR_the | TPR_a | TPR (C→M)
    
    where:
    - N: Total TPR observations for child speaker
    - TPR (M→C): Child's overall TPR (responding to mother)
    - TPR_the: Child's P(use "the" | mother used "a")
    - TPR_a: Child's P(use "a" | mother used "the")
    - TPR (C→M): Mother's overall TPR (responding to child)
    
    Args:
        per_session_results: DataFrame with per-session results
        
    Returns:
        DataFrame with aggregated results per child
    """
    aggregated = []
    
    for child_name in per_session_results['child_name'].unique():
        child_data = per_session_results[per_session_results['child_name'] == child_name]
        
        # Separate by speaker
        child_speaker = child_data[child_data['speaker'] == 'child']
        mother_speaker = child_data[child_data['speaker'] == 'mother']
        
        # Aggregate child's TPR (M→C)
        child_total = child_speaker['n_total'].sum()
        child_trans_the = child_speaker['n_transitions_to_the'].sum()
        child_trans_a = child_speaker['n_transitions_to_a'].sum()
        child_prev_a = child_speaker['n_previous_a'].sum()
        child_prev_the = child_speaker['n_previous_the'].sum()
        
        child_tpr = (child_trans_the + child_trans_a) / child_total if child_total > 0 else None
        child_tpr_the = child_trans_the / child_prev_a if child_prev_a > 0 else None
        child_tpr_a = child_trans_a / child_prev_the if child_prev_the > 0 else None
        
        # Aggregate mother's TPR (C→M)
        mother_total = mother_speaker['n_total'].sum()
        mother_trans_the = mother_speaker['n_transitions_to_the'].sum()
        mother_trans_a = mother_speaker['n_transitions_to_a'].sum()
        
        mother_tpr = (mother_trans_the + mother_trans_a) / mother_total if mother_total > 0 else None
        
        aggregated.append({
            'Dyad': child_name,
            'N': int(child_total) if child_total > 0 else 0,
            'TPR (M→C)': child_tpr,
            'TPR_the': child_tpr_the,
            'TPR_a': child_tpr_a,
            'TPR (C→M)': mother_tpr,
        })
    
    return pd.DataFrame(aggregated)


def prepare_cross_speaker_inputs_all_sessions(det_noun_locations: pd.DataFrame,
                                              output_dir: str,
                                              cache_filename: str,
                                              attested_nouns: Dict[str, Dict[str, set]] = None,
                                              include_without_context: bool = False,
                                              discourse_label_style: str = 'spoken') -> pd.DataFrame:
    """
    Pre-compute formatted inputs tracking cross-speaker references for ALL sessions.
    
    This is the shared core function for both TPR and contextual experiments.
    Tracks when OTHER speaker previously mentioned the same noun, formats discourse context.
    
    Args:
        det_noun_locations: All det-noun locations
        output_dir: Base output directory for caching
        cache_filename: Name of cache file (e.g., 'tpr_cached_inputs.csv')
        attested_nouns: Optional Dict[child_name][filename] -> Set[attested nouns].
                       If provided, filters to only these nouns (TPR mode).
                       If None, processes all nouns (contextual mode).
        include_without_context: If True, includes pairs even when OTHER speaker hasn't
                                mentioned noun yet (contextual mode). If False, only
                                includes pairs with prior mentions (TPR mode).
        discourse_label_style: 'spoken' (MOTHER/CHILD) or 'childes' (*MOT/*CHI)
        
    Returns:
        DataFrame with pre-formatted inputs for all sessions
    """
    # Check for cached inputs
    cache_file = os.path.join(output_dir, cache_filename)
    
    if os.path.exists(cache_file):
        print(f"  Loading cached inputs from {cache_file}")
        return pd.read_csv(cache_file)
    
    print("  Computing formatted inputs for all sessions...")
    all_prepared_inputs = []
    
    # Get all unique (child, filename) combinations
    if attested_nouns is not None:
        # TPR mode: only process children and sessions with attested nouns
        sessions_to_process = [
            (child_name, filename)
            for child_name in attested_nouns.keys()
            for filename in attested_nouns[child_name].keys()
            if len(attested_nouns[child_name][filename]) > 0
        ]
    else:
        # Contextual mode: process all sessions
        sessions_to_process = det_noun_locations.groupby(['child_name', 'filename']).groups.keys()
    
    # Process all children and sessions
    for child_name, filename in sessions_to_process:
        # Filter to this session
        session_filter = (
            (det_noun_locations['child_name'] == child_name) &
            (det_noun_locations['filename'] == filename)
        )
        
        # Apply noun filter if in TPR mode
        if attested_nouns is not None:
            session_attested = attested_nouns[child_name][filename]
            session_filter = session_filter & (det_noun_locations['noun'].isin(session_attested))
        
        session_locs = det_noun_locations[session_filter].copy().sort_values('line_num')
    
        session_locs = det_noun_locations[session_filter].copy().sort_values('line_num')
    
        # Track last mention by OTHER speaker for each noun (reset per session)
        last_mention_by_other = {}
        
        # Process chronologically to identify valid cross-speaker reference cases
        for idx, loc in session_locs.iterrows():
            current_speaker = loc['speaker_type']
            if pd.isna(current_speaker) or current_speaker not in ['child', 'mother', 'other_adult']:
                continue
            
            noun = loc['noun']
            det = loc['det']
            det_type = 'the' if det == 'the' else 'a'
            
            target_prior_speakers = []
            if current_speaker == 'child':
                target_prior_speakers = ['mother']
            elif current_speaker == 'mother':
                target_prior_speakers = ['child']
            elif current_speaker == 'other_adult':
                target_prior_speakers = ['mother']
            
            # Check if matching OTHER speaker mentioned this noun before
            has_prior_mention = False
            if noun in last_mention_by_other and last_mention_by_other[noun]['speaker'] in target_prior_speakers:
                has_prior_mention = True
            
            # Decide whether to include this case
            should_include = has_prior_mention or include_without_context
            
            if should_include:
                # Create masked sentences
                masked_sent_mlm_generic = (
                    loc['sentence'][:loc['det_char_start']] + 
                    '{MASK}' + 
                    loc['sentence'][loc['det_char_end']:]
                )
                
                masked_sent_ar = (
                    loc['sentence'][:loc['det_char_start']] + 
                    '___' + 
                    loc['sentence'][loc['det_char_end']:]
                )
                
                # Build entry based on whether there's prior mention
                entry = {
                    'sentence_id': idx,
                    'child_name': child_name,
                    'filename': os.path.basename(filename),
                    'speaker': current_speaker,
                    'noun': noun,
                    'original_sentence': loc['sentence'],
                    'original_det': det,
                    'masked_position': loc['det_token_idx'],
                    'target_line_num': loc['line_num'],
                    'det_char_start': loc['det_char_start'],
                    'det_char_end': loc['det_char_end']
                }
                
                if has_prior_mention:
                    prev_mention = last_mention_by_other[noun]
                    
                    # Extract all intervening utterances
                    intervening = det_noun_locations[
                        (det_noun_locations['filename'] == filename) &
                        (det_noun_locations['line_num'] > prev_mention['line_num']) &
                        (det_noun_locations['line_num'] < loc['line_num'])
                    ].copy()
                    
                    # Remove duplicates (same line may have multiple det-noun pairs)
                    intervening = intervening.drop_duplicates(subset=['line_num'])
                    intervening = intervening.sort_values('line_num')
                    
                    intervening_list = [{
                        'speaker_type': row['speaker_type'],
                        'sentence': row['sentence']
                    } for _, row in intervening.iterrows()]
                    
                    # Format inputs with prior mention as context
                    full_input_mlm, _ = format_discourse_input(
                        [prev_mention['sentence']],
                        current_speaker,
                        masked_sent_mlm_generic,
                        'mlm',
                        discourse_label_style=discourse_label_style
                    )
                    
                    full_input_ar, _ = format_discourse_input(
                        [prev_mention['sentence']],
                        current_speaker,
                        masked_sent_ar,
                        'ar',
                        discourse_label_style=discourse_label_style
                    )
                    
                    entry.update({
                        'previous_det': prev_mention['det'],
                        'previous_sentence': prev_mention['sentence'],
                        'previous_speaker': prev_mention['speaker'],
                        'previous_line_num': prev_mention['line_num'],
                        'intervening_utterances_json': json.dumps(intervening_list),
                        'full_input_mlm': full_input_mlm,
                        'full_input_ar': full_input_ar,
                    })
                    
                    # Add has_prior_mention flag if in contextual mode
                    if include_without_context:
                        entry['has_prior_mention'] = True
                else:
                    # No prior mention - use simple masked input
                    entry.update({
                        'previous_det': None,
                        'previous_sentence': None,
                        'previous_speaker': None,
                        'previous_line_num': None,
                        'intervening_utterances_json': '[]',
                        'full_input_mlm': masked_sent_mlm_generic,
                        'full_input_ar': masked_sent_ar,
                        'has_prior_mention': False
                    })
                
                all_prepared_inputs.append(entry)
            
            # Update last mention for this noun (by current speaker)
            last_mention_by_other[noun] = {
                'det': det_type,
                'speaker': current_speaker,
                'line_num': loc['line_num'],
                'sentence': loc['sentence']
            }
    
    # Save to cache
    if all_prepared_inputs:
        df = pd.DataFrame(all_prepared_inputs)
        df.to_csv(cache_file, index=False)
        print(f"  Saved {len(df)} formatted inputs to {cache_file}")
        if include_without_context:
            # Contextual mode - report breakdown
            with_context = df.get('has_prior_mention', pd.Series([False])).sum()
            without_context = len(df) - with_context
            print(f"  - With prior mention: {with_context}")
            print(f"  - Without prior mention: {without_context}")
        return df
    
    return pd.DataFrame()


def prepare_tpr_inputs_all_sessions(det_noun_locations: pd.DataFrame,
                                     attested_nouns: Dict[str, Dict[str, set]],
                                     tpr_output_dir: str,
                                     discourse_label_style: str = 'spoken') -> pd.DataFrame:
    """
    Pre-compute all formatted TPR inputs for ALL sessions and cache in one file.
    
    TPR mode: Only processes nouns child used with BOTH determiners, only includes
    cases where OTHER speaker mentioned the noun before.
    
    Args:
        det_noun_locations: All det-noun locations
        attested_nouns: Dict[child_name][filename] -> Set[attested nouns]
        tpr_output_dir: Base output directory for caching
        
    Returns:
        DataFrame with pre-formatted inputs for all sessions
    """
    return prepare_cross_speaker_inputs_all_sessions(
        det_noun_locations=det_noun_locations,
        output_dir=tpr_output_dir,
        cache_filename='tpr_cached_inputs.csv',
        attested_nouns=attested_nouns,
        include_without_context=False,  # TPR: only cases with prior mentions
        discourse_label_style=discourse_label_style
    )


def prepare_contextual_inputs_all_sessions(det_noun_locations: pd.DataFrame,
                                           contextual_output_dir: str) -> pd.DataFrame:
    """
    Pre-compute all formatted contextual inputs for ALL sessions and cache in one file.
    
    Contextual mode: Processes ALL det-noun pairs (no attested filter), but ONLY includes
    cases where OTHER speaker mentioned the noun before. Every input includes that prior
    mention plus intervening utterances.
    
    Difference from TPR: Processes ALL nouns (not just attested), giving more observations
    while maintaining the cross-speaker discourse context.
    
    Args:
        det_noun_locations: All det-noun locations
        contextual_output_dir: Base output directory for caching
        
    Returns:
        DataFrame with pre-formatted inputs for all sessions
    """
    return prepare_cross_speaker_inputs_all_sessions(
        det_noun_locations=det_noun_locations,
        output_dir=contextual_output_dir,
        cache_filename='contextual_cached_inputs.csv',
        attested_nouns=None,  # Contextual: process all nouns (not just attested)
        include_without_context=False  # Contextual: ONLY cases with OTHER speaker's prior mention
    )



def calculate_tpr_from_model_predictions(predictions_df: pd.DataFrame,
                                        det_noun_locations: pd.DataFrame,
                                        attested_nouns: set) -> Dict[str, Dict]:
    """
    Calculate TPR from model predictions for one session.
    
    Args:
        predictions_df: Predictions for one speaker in one session
        det_noun_locations: All det-noun locations
        attested_nouns: Set of attested nouns for this session
        
    Returns:
        Dict with TPR statistics
    """
    if len(predictions_df) == 0:
        return {
            'tpr_the': None,
            'tpr_a': None,
            'tpr_overall': None,
            'n_transitions_to_the': 0,
            'n_transitions_to_a': 0,
            'n_previous_a': 0,
            'n_previous_the': 0,
            'n_maintained': 0,
            'n_total': 0
        }
    
    # Count transitions based on predictions
    results = {
        'transitions_to_the': 0,
        'transitions_to_a': 0,
        'previous_a': 0,
        'previous_the': 0,
        'n_maintained': 0
    }
    
    for _, pred in predictions_df.iterrows():
        prev_det = pred['previous_det']
        predicted_det = pred['predicted_det_argmax']
        
        if prev_det == 'a':
            results['previous_a'] += 1
            if predicted_det == 'the':
                results['transitions_to_the'] += 1
            else:
                results['n_maintained'] += 1
        else:  # prev_det == 'the'
            results['previous_the'] += 1
            if predicted_det == 'a':
                results['transitions_to_a'] += 1
            else:
                results['n_maintained'] += 1
    
    n_total = results['previous_a'] + results['previous_the']
    tpr_the = results['transitions_to_the'] / results['previous_a'] if results['previous_a'] > 0 else None
    tpr_a = results['transitions_to_a'] / results['previous_the'] if results['previous_the'] > 0 else None
    tpr_overall = (results['transitions_to_the'] + results['transitions_to_a']) / n_total if n_total > 0 else None
    
    return {
        'tpr_the': tpr_the,
        'tpr_a': tpr_a,
        'tpr_overall': tpr_overall,
        'n_transitions_to_the': results['transitions_to_the'],
        'n_transitions_to_a': results['transitions_to_a'],
        'n_previous_a': results['previous_a'],
        'n_previous_the': results['previous_the'],
        'n_maintained': results['n_maintained'],
        'n_total': n_total
    }


def calculate_tpr_from_discourse_predictions(discourse_output_dir: str, 
                                             tpr_metadata_path: str,
                                             output_dir: str):
    """
    Calculate TPR metrics using Discourse experiment predictions matched with TPR metadata.
    
    This answers: "How do models perform on TPR-style cross-speaker cases when given
    ALL prior discourse (Experiment 2) vs. ONLY the other speaker's mention (Experiment 3)?"
    
    Strategy:
    1. Load TPR metadata (identifies valid cross-speaker reference cases)
    2. For each model in discourse outputs, match predictions to TPR cases
    3. Calculate TPR metrics (switch rates, accuracy, overlap)
    4. Save results in same format as TPR experiment
    
    Args:
        discourse_output_dir: Path to discourse experiment outputs (e.g., './output/manchester_discourse')
        tpr_metadata_path: Path to TPR cached inputs (e.g., './output/manchester_tpr/tpr_cached_inputs.csv')
        output_dir: Output directory for TPR metrics (e.g., './output/manchester_discourse/man_discourse_tpr')
    """
    import pandas as pd
    import os
    import json
    from pathlib import Path
    
    print("\n" + "="*80)
    print("CALCULATING TPR METRICS FROM DISCOURSE PREDICTIONS")
    print("="*80 + "\n")
    
    # Load TPR metadata (ground truth for valid cross-speaker cases)
    print(f"Loading TPR metadata from {tpr_metadata_path}...")
    tpr_metadata = pd.read_csv(tpr_metadata_path)
    print(f"  Loaded {len(tpr_metadata)} TPR cases")
    
    # Load model configs to map base names to full huggingface paths
    config_file = os.path.join(os.path.dirname(discourse_output_dir), '..', 'model_configs.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            model_configs = json.load(f)
            # Create mapping from base name to full name
            name_mapping = {}
            for config in model_configs:
                full_name = config.get('name', '')
                base_name = full_name.split('/')[-1] if '/' in full_name else full_name
                name_mapping[base_name] = full_name
        print(f"  Loaded {len(name_mapping)} model name mappings")
    else:
        print(f"  Warning: Config file not found at {config_file}, will use directory names")
        name_mapping = {}
    
    
    # Create matching key for TPR metadata
    tpr_metadata['match_key'] = (
        tpr_metadata['child_name'] + '|' + 
        tpr_metadata['speaker'] + '|' + 
        tpr_metadata['noun'] + '|' + 
        tpr_metadata['original_sentence']
    )
    
    # Create lookup for TPR metadata
    tpr_lookup = tpr_metadata.set_index('match_key')
    
    # Find all discourse prediction files from supported model-type directories.
    discourse_path = Path(discourse_output_dir)

    # Only process canonical model family directories.
    # This intentionally excludes archival folders (e.g., *_OLD, left-context-only).
    allowed_dirs = ['ar', 'mlm', 'seq2seq']
    prediction_files = []
    for model_type_dir in allowed_dirs:
        type_path = discourse_path / model_type_dir
        if type_path.exists():
            prediction_files.extend(list(type_path.glob('*/*/child_predictions.csv')))
            prediction_files.extend(list(type_path.glob('*/*/mother_predictions.csv')))
    
    print(f"\nFound {len(prediction_files)} discourse prediction files (from {', '.join(allowed_dirs)} directories)")
    
    # Normalize sentence function to handle tokenization differences
    def normalize_sentence(s):
        """Remove extra spaces around punctuation for consistent matching."""
        import re
        # Remove spaces before punctuation
        s = re.sub(r'\s+([.,!?;:])', r'\1', s)
        # Normalize multiple spaces to single space
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
    
    # Create lookup key using sentence_id (unique identifier from corpus)
    # This is more reliable than matching on sentence text
    tpr_metadata['match_key'] = (
        tpr_metadata['child_name'] + '|' + 
        tpr_metadata['speaker'] + '|' + 
        tpr_metadata['sentence_id'].astype(str)
    )
    tpr_lookup = tpr_metadata.set_index('match_key')
    
    # Group by model
    from collections import defaultdict
    model_files = defaultdict(lambda: defaultdict(list))
    
    for pred_file in prediction_files:
        speaker_type = 'child' if pred_file.name == 'child_predictions.csv' else 'mother'
        child_name = pred_file.parent.name
        model_name = pred_file.parent.parent.name
        model_type = pred_file.parent.parent.parent.name
        
        model_key = f"{model_type}/{model_name}"
        model_files[model_key][(child_name, speaker_type)].append(pred_file)
    
    print(f"Processing {len(model_files)} unique models\n")
    
    # Process each model
    all_results = []
    
    for model_key, child_speaker_files in model_files.items():
        model_type, model_name_base = model_key.split('/')
        
        # Get full model name from mapping if available
        full_model_name = name_mapping.get(model_name_base, model_name_base)
        
        print(f"\nProcessing {full_model_name} ({model_type.upper()})...")
        
        model_matched = []
        model_unmatched = 0
        
        # Process each child/speaker combination
        for (child_name, speaker_type), files in child_speaker_files.items():
            for pred_file in files:
                # Load discourse predictions
                discourse_preds = pd.read_csv(pred_file)
                
                # Use the full model name from our mapping (preserves huggingface path)
                pred_model_name = full_model_name
                
                # Create matching key using sentence_id (unique corpus identifier)
                discourse_preds['match_key'] = (
                    discourse_preds['child_name'] + '|' + 
                    discourse_preds['speaker'] + '|' + 
                    discourse_preds['sentence_id'].astype(str)
                )
                
                # Match with TPR metadata
                for idx, row in discourse_preds.iterrows():
                    if row['match_key'] in tpr_lookup.index:
                        tpr_info = tpr_lookup.loc[row['match_key']]
                        
                        # sentence_id should be unique, but handle edge cases
                        if isinstance(tpr_info, pd.DataFrame):
                            if len(tpr_info) > 1:
                                print(f"  Warning: Multiple TPR matches for {row['match_key']}, using first")
                            tpr_info = tpr_info.iloc[0]
                        
                        # Combine discourse prediction with TPR metadata
                        matched_row = {
                            'model_name': pred_model_name,  # Use model_name from predictions (full path)
                            'model_type': model_type,
                            'child_name': row['child_name'],
                            'filename': str(tpr_info['filename']),
                            'speaker': row['speaker'],
                            'noun': row['noun'],
                            'original_sentence': row['original_sentence'],
                            'original_det': row['original_det'],
                            'previous_det': str(tpr_info['previous_det']),
                            'previous_speaker': str(tpr_info['previous_speaker']),
                            'P_a': row['P_a'],
                            'P_the': row['P_the'],
                            'predicted_det_argmax': row['predicted_det_argmax'],
                            'predicted_det_sample': row['predicted_det_sample'],
                        }
                        model_matched.append(matched_row)
                    else:
                        model_unmatched += 1
        
        print(f"  Matched: {len(model_matched)} cases")
        print(f"  Unmatched: {model_unmatched} cases (not in TPR metadata)")
        
        if len(model_matched) > 0:
            # Save matched predictions for this model
            matched_df = pd.DataFrame(model_matched)
            
            # Extract the full model_name from matched_df (all rows should have the same model_name)
            full_model_name_from_df = matched_df['model_name'].iloc[0]
            
            # Create output directory (use base name for directory structure)
            model_output_dir = os.path.join(output_dir, model_type, model_name_base)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Save full matched predictions
            matched_path = os.path.join(model_output_dir, 'matched_tpr_cases.csv')
            matched_df.to_csv(matched_path, index=False)
            print(f"  Saved matched cases to {matched_path}")
            
            # Calculate TPR metrics per session first, then aggregate
            # This matches the structure of the original TPR experiment
            for child_name in matched_df['child_name'].unique():
                child_subset = matched_df[matched_df['child_name'] == child_name]
                for filename in child_subset['filename'].unique():
                    filename_str = str(filename)  # Ensure scalar
                    for speaker_type in ['child', 'mother']:
                        subset = child_subset[
                            (child_subset['filename'] == filename_str) &
                            (child_subset['speaker'] == speaker_type)
                        ]
                        
                        if len(subset) > 0:
                            # Calculate for both deterministic and probabilistic methods.
                            for method_label, pred_col in [('det', 'predicted_det_argmax'), ('prob', 'predicted_det_sample')]:
                                
                                # Calculate TPR metrics for this session
                                metrics = calculate_tpr_metrics_for_predictions(
                                    subset, 
                                    pred_col
                                )
                                
                                metrics.update({
                                    'model_name': full_model_name_from_df,  # Use full model_name from predictions
                                    'model_type': model_type,
                                    'child_name': child_name,
                                    'filename': filename_str,
                                    'speaker': speaker_type,
                                    'method': method_label,
                                    'source': 'model'
                                })
                                
                                all_results.append(metrics)
    
    # Save aggregated results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save per-session results (legacy filename)
        per_session_path = os.path.join(output_dir, 'tpr_metrics_per_session.csv')
        results_df.to_csv(per_session_path, index=False)
        print(f"\n✓ Saved per-session TPR metrics to {per_session_path}")
        print(f"  Total: {len(results_df)} session/model/speaker/method combinations")

        # Save Manchester-TPR-compatible long-form file name
        # Includes an extra 'method' column because discourse-derived results include argmax/sample.
        tpr_all_path = os.path.join(output_dir, 'tpr_all_results.csv')
        results_df.to_csv(tpr_all_path, index=False)
        print(f"✓ Saved Manchester-compatible long-form TPR file to {tpr_all_path}")
        
        # Also save aggregated summary (summing across sessions per child)
        aggregated = []
        for model_name in results_df['model_name'].unique():
            for model_type in results_df[results_df['model_name'] == model_name]['model_type'].unique():
                for child_name in results_df['child_name'].unique():
                    for speaker in ['child', 'mother']:
                        for method in ['det', 'prob']:
                            subset = results_df[
                                (results_df['model_name'] == model_name) &
                                (results_df['model_type'] == model_type) &
                                (results_df['child_name'] == child_name) &
                                (results_df['speaker'] == speaker) &
                                (results_df['method'] == method)
                            ]
                            
                            if len(subset) > 0:
                                # Aggregate across sessions
                                n_total = subset['n_total'].sum()
                                n_correct = subset['n_correct'].sum()
                                n_prev_a = subset['n_previous_a'].sum()
                                n_prev_the = subset['n_previous_the'].sum()
                                n_maintained = subset['n_maintained'].sum() if 'n_maintained' in subset.columns else 0
                                
                                # Recalculate rates from aggregated counts
                                accuracy = n_correct / n_total if n_total > 0 else None
                                
                                # Recalculate rates from explicit transition counts.
                                # This avoids NaN propagation from per-session rate columns.
                                n_transitions_to_the = subset['n_transitions_to_the'].sum() if 'n_transitions_to_the' in subset.columns else 0
                                n_transitions_to_a = subset['n_transitions_to_a'].sum() if 'n_transitions_to_a' in subset.columns else 0
                                
                                tpr_the = n_transitions_to_the / n_prev_a if n_prev_a > 0 else None
                                tpr_a = n_transitions_to_a / n_prev_the if n_prev_the > 0 else None
                                tpr_overall = (n_transitions_to_the + n_transitions_to_a) / n_total if n_total > 0 else None
                                
                                aggregated.append({
                                    'model_name': model_name,
                                    'model_type': model_type,
                                    'child_name': child_name,
                                    'speaker': speaker,
                                    'method': method,
                                    'source': 'model',
                                    'n_attested_nouns': np.nan,
                                    'n_total': int(n_total),
                                    'n_previous_a': int(n_prev_a),
                                    'n_previous_the': int(n_prev_the),
                                    'n_transitions_to_the': int(n_transitions_to_the),
                                    'n_transitions_to_a': int(n_transitions_to_a),
                                    'n_maintained': int(n_maintained),
                                    'tpr_the': tpr_the,
                                    'tpr_a': tpr_a,
                                    'tpr_overall': tpr_overall,
                                    # Backward-compatible aliases for existing notebook cell
                                    'switch_rate_after_a': tpr_the,
                                    'switch_rate_after_the': tpr_a,
                                    'overall_switch_rate': tpr_overall,
                                    'accuracy': accuracy,
                                    'n_correct': int(n_correct),
                                    'n_sessions': len(subset)
                                })
        
        if aggregated:
            agg_df = pd.DataFrame(aggregated)
            summary_path = os.path.join(output_dir, 'tpr_metrics_aggregated.csv')
            agg_df.to_csv(summary_path, index=False)
            print(f"✓ Saved aggregated TPR metrics to {summary_path}")
            print(f"  Total: {len(agg_df)} child/model/speaker/method combinations")
    
    print("\n" + "="*80)
    print("TPR CALCULATION FROM DISCOURSE COMPLETE")
    print("="*80 + "\n")


def calculate_tpr_metrics_for_predictions(matched_df: pd.DataFrame, 
                                          pred_col: str) -> dict:
    """
    Calculate TPR metrics from matched discourse predictions.
    
    Args:
        matched_df: DataFrame with both predictions and TPR metadata
        pred_col: Column name for predictions ('predicted_det_argmax' or 'predicted_det_sample')
    
    Returns:
        Dictionary with TPR metrics
    """
    # Count by previous determiner
    prev_a = matched_df[matched_df['previous_det'] == 'a']
    prev_the = matched_df[matched_df['previous_det'] == 'the']
    
    # Transition counts following Manchester TPR naming
    n_transitions_to_the = (prev_a[pred_col] == 'the').sum() if len(prev_a) > 0 else 0
    n_transitions_to_a = (prev_the[pred_col] == 'a').sum() if len(prev_the) > 0 else 0

    if len(prev_a) > 0:
        tpr_the = n_transitions_to_the / len(prev_a)
    else:
        tpr_the = None
    
    if len(prev_the) > 0:
        tpr_a = n_transitions_to_a / len(prev_the)
    else:
        tpr_a = None
    
    # Overall switch rate
    if len(matched_df) > 0:
        switched = (matched_df[pred_col] != matched_df['previous_det']).sum()
        tpr_overall = switched / len(matched_df)
    else:
        tpr_overall = None
    
    # Accuracy (matches human original_det)
    if len(matched_df) > 0:
        correct = (matched_df[pred_col] == matched_df['original_det']).sum()
        accuracy = correct / len(matched_df)
    else:
        accuracy = None
    
    # Maintenance (matches previous_det)
    if len(matched_df) > 0:
        maintained = (matched_df[pred_col] == matched_df['previous_det']).sum()
        maintenance_rate = maintained / len(matched_df)
    else:
        maintenance_rate = None
    
    return {
        'n_attested_nouns': np.nan,
        'n_total': len(matched_df),
        'n_previous_a': len(prev_a),
        'n_previous_the': len(prev_the),
        'n_transitions_to_the': int(n_transitions_to_the),
        'n_transitions_to_a': int(n_transitions_to_a),
        'n_maintained': int(maintained) if len(matched_df) > 0 else 0,
        'tpr_the': tpr_the,
        'tpr_a': tpr_a,
        'tpr_overall': tpr_overall,
        # Backward-compatible aliases
        'switch_rate_after_a': tpr_the,
        'switch_rate_after_the': tpr_a,
        'overall_switch_rate': tpr_overall,
        'maintenance_rate': maintenance_rate,
        'accuracy': accuracy,
        'n_correct': correct if len(matched_df) > 0 else 0,
    }


if __name__ == '__main__':
    print("cac_utils.py loaded successfully")
    print(f"Device: {device}")
    print(f"Available functions: {[name for name in dir() if not name.startswith('_')]}")
