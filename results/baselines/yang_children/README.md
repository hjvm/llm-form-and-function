# Yang's six children (follow-up, not in the paper)

Yang (2011, CMCL, Table 1; the nine child samples of Yang 2013 PNAS are these six plus pooled first 100/300/500 pairs):
Adam, Eve, Sarah (CHILDES Brown), Naomi (Sachs), Nina (Suppes), Peter (Bloom). Put them under
`data/CHILDES/Yang/<Child>/` (Naomi = Sachs, Nina = Suppes) and run from the repo root:

    python scripts/yang_children/yang_extract.py Yang        # paper's loader and extractor
    python scripts/yang_children/yang_dxn.py                 # children's D×N with Carl replaced / Yang's children added
    python scripts/yang_children/yang_coins.py               # 0.82 and noun-level coins on the same samples

Extraction used spaCy 3.8 / en_core_web_sm 3.8.0 (3.7.1 does not install on Python 3.13). Re-extracting Carl with
3.8 gives S 4024 vs 4041 and emp/pred 0.401/0.517 vs 0.400/0.519. Samples are trimmed to Yang's age ranges
(`_trim`); `_full` uses every file. The paper's 49 LMs have not been run on these children.
