"""Transitional Probability of Reference (TPR) for human speakers, and TPR case metadata.

A TPR case is a D×N site whose noun the *other* speaker mentioned earlier in the same
session (child after mother, mother after child, other adult after mother). TPR is the
rate at which the speaker switches determiner relative to that antecedent
(Gleitman & Yang 2022).
"""
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Whose earlier mention counts as the antecedent for each speaker.
ANTECEDENT_SPEAKERS = {
    'child': ['mother'],
    'mother': ['child'],
    'other_adult': ['mother'],
}


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


def calculate_tpr_human_baseline_unrestricted(det_noun_locations: pd.DataFrame,
                                              output_path: str) -> pd.DataFrame:
    """
    Per-session human TPR over all extracted nouns (no attested-noun filter).

    This unrestricted noun set is the one used for every reported result.

    Args:
        det_noun_locations: site inventory of one corpus.
        output_path: CSV file to write the per-session results to.
    """
    all_results = []
    for child_name, child_locs in det_noun_locations.groupby("child_name", sort=False):
        print(f"  Calculating unrestricted TPR for {child_name}...")
        for filename, session_locs in child_locs.groupby("filename", sort=False):
            session_locs = session_locs.sort_values("line_num")
            session_tpr = calculate_tpr_per_session(session_locs)
            for speaker in ["child", "mother", "other_adult"]:
                all_results.append({
                    "child_name": child_name,
                    "filename": filename,
                    "speaker": speaker,
                    "source": "human",
                    "model_name": "Human",
                    "model_type": "baseline_unrestricted",
                    "n_unique_nouns_in_session": int(session_locs["noun"].nunique()),
                    **session_tpr[speaker],
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"  Saved unrestricted human baseline to {output_path}")
    return results_df


def prepare_tpr_cases(det_noun_locations: pd.DataFrame,
                      attested_nouns: Optional[Dict[str, Dict[str, set]]] = None) -> pd.DataFrame:
    """
    The TPR cases: one row per site whose noun was last mentioned, earlier in the same
    session, by the other speaker (child <- mother, mother <- child, other adult <- mother).

    The paper uses every noun (``attested_nouns=None``). Passing ``attested_nouns`` restricts
    each session to those nouns. ``sentence_id`` is the site's row in the inventory, which is
    how Step 3 joins model predictions to cases.
    """
    if attested_nouns is None:
        sessions = list(det_noun_locations.groupby(['child_name', 'filename']).groups.keys())
    else:
        sessions = [
            (child_name, filename)
            for child_name, sessions_of_child in attested_nouns.items()
            for filename, nouns in sessions_of_child.items()
            if len(nouns) > 0
        ]

    cases = []
    for child_name, filename in sessions:
        in_session = ((det_noun_locations['child_name'] == child_name)
                      & (det_noun_locations['filename'] == filename))
        if attested_nouns is not None:
            in_session &= det_noun_locations['noun'].isin(attested_nouns[child_name][filename])
        session_locs = det_noun_locations[in_session].sort_values('line_num')

        last_mention = {}  # noun -> most recent mention in this session
        for idx, loc in session_locs.iterrows():
            speaker = loc['speaker_type']
            if pd.isna(speaker) or speaker not in ANTECEDENT_SPEAKERS:
                continue
            noun = loc['noun']
            prev = last_mention.get(noun)
            if prev is not None and prev['speaker'] in ANTECEDENT_SPEAKERS[speaker]:
                cases.append({
                    'sentence_id': idx,
                    'child_name': child_name,
                    'filename': os.path.basename(filename),
                    'speaker': speaker,
                    'noun': noun,
                    'original_sentence': loc['sentence'],
                    'original_det': loc['det'],
                    'masked_position': loc['det_token_idx'],
                    'target_line_num': loc['line_num'],
                    'det_char_start': loc['det_char_start'],
                    'det_char_end': loc['det_char_end'],
                    'previous_det': prev['det'],
                    'previous_sentence': prev['sentence'],
                    'previous_speaker': prev['speaker'],
                    'previous_line_num': prev['line_num'],
                })
            last_mention[noun] = {
                'det': 'the' if loc['det'] == 'the' else 'a',
                'speaker': speaker,
                'line_num': loc['line_num'],
                'sentence': loc['sentence'],
            }

    return pd.DataFrame(cases)


def add_antecedent_distance(cases: pd.DataFrame, corpus: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add ``true_dist``: how many utterances back (within the session) the antecedent is.

    1 means the antecedent is the immediately preceding utterance. Used to check which
    TPR cases fall inside each model's context window.
    """
    streams = {}
    for child, df in corpus.items():
        for fn, g in df.groupby("filename"):
            streams[(child, os.path.basename(str(fn)))] = np.sort(g["line_num"].values)

    def _dist(r):
        arr = streams.get((r["child_name"], os.path.basename(str(r["filename"]))))
        if arr is None or pd.isna(r["previous_line_num"]):
            return np.nan
        return int(((arr > r["previous_line_num"]) & (arr < r["target_line_num"])).sum()) + 1

    cases = cases.copy()
    cases["true_dist"] = cases.apply(_dist, axis=1)
    return cases
