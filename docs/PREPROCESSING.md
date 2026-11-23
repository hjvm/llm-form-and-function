# Preprocessing across CHILDES corpora

The paper's loader (`src/determiner/corpus.py`) was built on the Manchester files and stays frozen:
the paper's Manchester and tpr-data site inventories are committed and are not re-extracted.
`src/determiner/preprocess.py` is the general version used for every child added after the paper
(Brown: Adam, Eve, Sarah; Bloom: Peter; Sachs: Naomi; Suppes: Nina; Lara; Providence: Naima, Lily,
Alex, William, Ethan).

## Design

Shared steps, every corpus:

1. **Read.** Join tab-continuation lines; keep each utterance's `%mor` tier (for auditing); take
   speaker roles from the `@ID` headers. `line_num` is the ordinal of the main tier in the file, so
   files with or without dependent tiers number utterances identically.
2. **Roles.** Target_Child → child, Mother → mother, adult roles (Father, Investigator, grandparents,
   Adult, Relative, Aunt/Uncle, Visitor, Family_Friend, ...) → other_adult. Siblings, playmates,
   friends, cousins, "Child", "Participant", Unidentified, Media (TV) and Environment are dropped.
3. **Clean** (`clean_utterance`), resolving CHAT to the intended words as CLAN's MOR reads them:
   replacements `de [: the]` → `the`; omitted sounds `(th)e`, `goin(g)`, `doll('s)` filled in;
   retraced material before `[/] [//] [///]` dropped; omitted words (`0is`, `&=0is`, `[* &=0is]`)
   inserted for the parser, **except a/an/the** (never create a determiner the speaker did not say);
   `xxx yyy www`, `&`-codes, `@`-suffixes (whole suffix, e.g. `@wp`, `@s:spa`), postcodes, pauses,
   terminators, media bullets, tag marks `„ ‡`, and Providence's `‹ ›` group marks removed.

Per-corpus step (`PROFILES`): file layout (one directory of sessions per child, or tpr-data's one file
per dyad with a session-id prefix), speaker-code overrides, and extra regex rules (tpr-data: flattened
`123_456` time codes).

## What the legacy loader got wrong outside Manchester

| Issue | Manchester | Elsewhere |
|---|---|---|
| Replacements kept the child's form (`de [: the]` → "de") | ~5 cases | Adam 1,678; Sarah 90; Naomi 20 |
| Omitted sounds cut words (`(th)e` → "e", `an(d)` → "an", `c(a)n` → "cn") | ~0.1% of utterances | 2–10% (Brown, Bloom, Providence) |
| Omitted words: `&=0is` inserted but `0is` deleted | Manchester notation (9,680) | Providence notation (3,000+), so treated differently |
| `&=0a`, `&=0the` inserted as determiners | 9 false sites | 3 |
| Overlap-marked speech `<a whale> [>]` deleted entirely | small | small |
| `yyy` not removed (regex typo `yyyy`) | none | Sarah 2%, Peter 3% of utterances |
| tpr-data: wrapped main-tier lines dropped | Anne 354 lines | Lara 1,322 lines |
| tpr-data: every non-CHI/MOT speaker = other adult | — | TV/media, siblings, child visitors |

The last row affects the paper's other-adult TPR reference: 56 of its 1,615 cases (3.5%) come from
media (37), sisters (7), children (6), environment (3) and playmates (3). Adult roles only: 0.2085
(n = 1,559) instead of 0.2155. The paper's value is kept as the primary reference; the adult-only
version is reported alongside.

## Extraction accuracy

Agreement of spaCy sites with each utterance's own `%mor` tier (CLAN MOR/POST, an independent tagger),
CHI and MOT utterances with `%mor`. A site matches when the determiner agrees and the noun lemmas are
equal or one ends the other (compounds are split differently). `%mor` is not gold: its old-style
lemmas (`n|dog-DIM` for "doggie", `adj|pot&dn-Y` for "potty", `n:let|h`) count against spaCy in every
corpus but Manchester, whose 2024 `%mor` is UD-style. Numbers are precision / recall.

| Corpus | Speaker | Sites (legacy) | Sites (shared) | Legacy P / R | Shared P / R |
|---|---|---|---|---|---|
| Manchester | child | 19,029 | 19,113 | .97 / .87 | .97 / .88 |
| Manchester | mother | 55,523 | 55,688 | .97 / .85 | .97 / .85 |
| Brown | child | 9,732 | 10,657 | .82 / .72 | .87 / .84 |
| Brown | mother | 11,370 | 11,378 | .89 / .84 | .89 / .84 |
| Bloom | child | 4,035 | 3,871 | .81 / .80 | .88 / .83 |
| Bloom | mother | 765 | 731 | .87 / .86 | .93 / .88 |
| Sachs | child | 1,270 | 1,302 | .89 / .82 | .91 / .86 |
| Sachs | mother | 1,619 | 1,627 | .87 / .81 | .87 / .81 |
| Suppes | child | 5,883 | 5,872 | .94 / .89 | .94 / .89 |
| Suppes | mother | 10,727 | 10,732 | .95 / .89 | .95 / .89 |
| Lara | child | 3,965 | 4,046 | .87 / .83 | .87 / .85 |
| Lara | mother | 10,654 | 10,776 | .86 / .82 | .86 / .83 |
| Providence | child | 12,364 | 13,464 | .82 / .71 | .88 / .83 |
| Providence | mother | 51,984 | 52,097 | .89 / .82 | .90 / .83 |

Most recall loss in every corpus is definitional (`%mor` counts "a bit", telegraphic noun pairs, and
letter names that spaCy does not tag as singular common nouns), and is the same for Manchester.

**Hand check** (20 random child sites per corpus, shared pipeline; `results/preprocessing_audit/hand_check_sample.csv`
has a further sample with `%mor` disagreements):

| Corpus | Correct | Errors |
|---|---|---|
| Manchester (paper inventory) | 20/20 | — |
| Brown | 18/20 | "trying a tell you" (a = to); "a wheel wowheel@b" (babble taken as noun) |
| Bloom | 20/20 | — |
| Sachs | 19/20 | "a good xxx" → "a good" (adjective as noun once xxx is removed) |
| Suppes | 20/20 | — |
| Lara | 18/20 | "what's a [*] happening"; "the (a)nother bike" |
| Providence | 17/20 | "a making this house"; "a pretend Emma"; letter name "a e@l" |

Residual errors are child-grammar cases (a for to, a + verb) and letter names, which no cleaning
rule fixes.

Scripts: `scripts/preprocessing_audit/audit_extraction.py` (scores both pipelines),
`summarize_audit.py`, `sample_for_hand_check.py`, `extract_new_children.py` (new inventories →
`results/baselines/all_children/{yang,tpr6}_det_noun_locations.csv.gz`).

## Adult TPR reference

**Main reference: adults talking with each other.** Every transition in which an adult (mother or
other adult, adult roles only) uses a D×N noun last mentioned earlier in the same session by a
*different* adult: the mother following another adult, other adults following the mother, and other
adults following each other. All 24 dyads, shared preprocessing (Manchester re-extracted with all
speakers for this purpose only). Pooled switch rate **.2221** (n = 4,989; 23 dyads with ≥10 cases,
dyad rates .15–.30). `scripts/yang_children/adult_tpr_reference.py` → `results/baselines/all_children/adult_tpr_reference.*`.

The paper's .2155 used a narrower set: other adults following the mother only, in the 17 tpr-data
dyads, with TV/media, siblings and child visitors counted as adults. Other definitions reported for
comparison: other adult following the mother, 24 dyads, .2048 (n = 2,109); plus the mother following
another adult, .2212 (n = 4,074).

Mothers switch more when following another adult (.239, n = 1,965) than when following their child
(.203 per-dyad mean). Under the main reference the 24 caretakers are borderline (p = .049); a
leave-one-dyad-out reference gives the same (p = .052), so it is not the mothers' own transitions
inflating the reference. `scripts/yang_children/adult_reference_loo.py`.
