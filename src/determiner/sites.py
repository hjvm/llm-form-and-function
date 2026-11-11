"""The D×N site inventory: every singular `a/an/the + noun` phrase in the corpus.

Each site is one determiner–noun phrase at a fixed character offset in a transcript
utterance. The inventory defines what both benchmarks evaluate, for humans and models.

The paper's inventories were extracted with spaCy ``en_core_web_sm`` 3.7.1. Newer spaCy
models tag the same sentences differently, so they are committed under
``results/site_inventory/`` and copied into the pipeline caches instead of re-extracted.
Re-extraction from scratch reproduces them only under ``spacy>=3.7,<3.8`` with
``en_core_web_sm==3.7.1``.
"""
import os
import shutil

import pandas as pd

DET_NOUN_LOCATIONS = 'det_noun_locations.csv'

# Relative to the repository root (see determiner.paths.chdir_to_root).
SITE_INVENTORY_DIR = './results/site_inventory'
# Cache directories whose name contains the key are seeded from the committed file.
SITE_INVENTORY_SEEDS = {
    'manchester': 'manchester_det_noun_locations.csv',
    'tpr_data':   'tpr_data_det_noun_locations.csv',
}

SPACY_MODEL = 'en_core_web_sm'
MAX_GAP = 1           # at most one token (e.g. an adjective) between determiner and noun
PIPE_BATCH_SIZE = 1000


def extract_det_noun_locations(corpus, output_dir):
    """
    Extract every singular D×N site from the corpus with spaCy and save it to
    ``<output_dir>/det_noun_locations.csv``.

    A site is a noun chunk that starts with a/an/the, whose head is a singular NOUN at
    most ``MAX_GAP`` tokens after the determiner.

    Args:
        corpus: {child_name: DataFrame} from a corpus loader.
        output_dir: cache directory to write the inventory into.

    Returns:
        DataFrame with columns child_name, filename, line_num, speaker, speaker_type,
        sentence, det, noun, pair_index, det_char_start, det_char_end, det_token_idx,
        noun_token_idx.
    """
    import spacy

    print("Extracting determiner-noun pairs from corpus...")
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])

    all_locations = []
    for child_name, child_df in corpus.items():
        print(f"  Processing {child_name}...")
        docs = nlp.pipe(child_df['sentence'].tolist(), batch_size=PIPE_BATCH_SIZE)

        for doc, (_, row) in zip(docs, child_df.iterrows()):
            pair_idx = 0
            for chunk in doc.noun_chunks:
                det_token = chunk[0]
                if det_token.pos_ != "DET" or det_token.lower_ not in ("a", "an", "the"):
                    continue
                gap = chunk.root.i - det_token.i - 1
                if not 0 <= gap <= MAX_GAP:
                    continue
                if chunk.root.pos_ != "NOUN" or "Sing" not in chunk.root.morph.get("Number"):
                    continue
                all_locations.append({
                    'child_name': child_name,
                    'filename': row['filename'],
                    'line_num': row['line_num'],
                    'speaker': row['speaker'],
                    'speaker_type': row['speaker_type'],
                    'sentence': row['sentence'],
                    'det': "a" if det_token.lower_ in ("a", "an") else "the",
                    'noun': chunk.root.lemma_.lower(),
                    'pair_index': pair_idx,
                    'det_char_start': det_token.idx,
                    'det_char_end': det_token.idx + len(det_token.text),
                    'det_token_idx': det_token.i,
                    'noun_token_idx': chunk.root.i,
                })
                pair_idx += 1

    locations_df = pd.DataFrame(all_locations)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, DET_NOUN_LOCATIONS)
    locations_df.to_csv(output_file, index=False)
    print(f"Saved {len(locations_df)} determiner-noun pairs to {output_file}")
    return locations_df


def load_or_extract_det_noun_locations(corpus, output_dir):
    """
    Return the site inventory for ``output_dir``, in order of preference:

    1. the cached ``det_noun_locations.csv`` already in ``output_dir``;
    2. a copy of the committed inventory in ``results/site_inventory/``;
    3. a fresh spaCy extraction (warns: only reproduces the paper under spaCy 3.7.1).

    Args:
        corpus: {child_name: DataFrame} from a corpus loader (used only for extraction).
        output_dir: cache directory for this corpus.
    """
    locations_file = os.path.join(output_dir, DET_NOUN_LOCATIONS)

    if not os.path.exists(locations_file):
        dir_name = os.path.basename(os.path.normpath(output_dir)).lower()
        for key, seed_name in SITE_INVENTORY_SEEDS.items():
            seed_file = os.path.join(SITE_INVENTORY_DIR, seed_name)
            if key in dir_name and os.path.exists(seed_file):
                print(f"Seeding det-noun pairs from committed inventory {seed_file}")
                os.makedirs(output_dir, exist_ok=True)
                shutil.copyfile(seed_file, locations_file)
                break

    if os.path.exists(locations_file):
        print(f"Loading det-noun pairs from {locations_file}")
        locations_df = pd.read_csv(locations_file)
        print(f"Loaded {len(locations_df)} determiner-noun pairs")
        return locations_df

    print("No site inventory found. Extracting from corpus...")
    print(f"WARNING: extraction is spaCy-version dependent; reproducing the published sites "
          f"requires {SPACY_MODEL}==3.7.1 (spacy>=3.7,<3.8).")
    return extract_det_noun_locations(corpus, output_dir)
