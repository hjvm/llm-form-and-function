"""Loading and cleaning CHILDES transcripts (Manchester and the tpr-data adult corpus)."""
import glob
import os
import re
from typing import Dict

import pandas as pd

# Corpus paths are kept relative to the repository root on purpose: the loader stores
# each transcript's path in the ``filename`` column, and the committed site inventories
# (results/site_inventory/) key every evaluation site by that exact string.
DEFAULT_CORPUS_PATH = './data/CHILDES/Manchester'
TPR_DATA_CORPUS_PATH = 'data/CHILDES/tpr-data'


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


def load_tpr_data_corpus(data_dir: str = TPR_DATA_CORPUS_PATH) -> Dict[str, pd.DataFrame]:
    """
    Load the tpr-data corpus (17 dyads with other-adult speakers; the adult TPR reference).

    Each ``.cha`` file holds one dyad; every utterance line is ``<session_id> <*SPK:> <text>``.
    ``*CHI`` and ``*MOT`` map to child and mother; every other speaker is ``other_adult``.

    Returns:
        dict: {dyad_name: DataFrame} with columns
        ['filename', 'speaker', 'sentence', 'line_num', 'speaker_type'], where
        ``filename`` is the session id.
    """
    corpus_data = {}
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.cha"))):
        child_name = os.path.basename(filepath).replace(".cha", "")

        rows = []
        line_num = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(maxsplit=2)
                if len(parts) >= 3 and parts[1].startswith("*"):
                    session_id, speaker, sentence = parts
                    if speaker == "*CHI:":
                        speaker_type = "child"
                    elif speaker == "*MOT:":
                        speaker_type = "mother"
                    else:
                        speaker_type = "other_adult"
                    rows.append({
                        "filename": session_id,
                        "speaker": speaker,
                        "sentence": sentence,
                        "line_num": line_num,
                        "speaker_type": speaker_type,
                    })
                line_num += 1

        if rows:
            df = pd.DataFrame(rows)
            df["sentence"] = df["sentence"].apply(clean_childes_annotations)
            df = df[df["sentence"].str.strip() != ""].copy()
            corpus_data[child_name] = df

            counts = df["speaker_type"].value_counts()
            print(f"Loaded {child_name}: {len(df)} utterances ({counts.get('child', 0)} chi, "
                  f"{counts.get('mother', 0)} mot, {counts.get('other_adult', 0)} other)")

    return corpus_data
