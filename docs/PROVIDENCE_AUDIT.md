# Providence children and the D×N test (Nov 24)

**Question.** Charles reports that the Providence children pass the D×N test with the Goldin-Meadow & Yang (2017) method. In our pipeline they over-overlap (empirical > predicted) and fail. Why?

> **Update Nov 25 (section 5):** Charles's code (Yang & Valian 2020 repo) uses the **same per-noun bias** as the paper, not 2/3. The 2/3 below only reproduces Yang's 2013 table. Under his code, the Providence children still over-overlap in every variant. See section 5.

**Short answer (Nov 24).** The det–noun lists are fine. The difference is the **bias** that goes into the predicted overlap. Yang's published expected values all correspond to a bias of about **2/3** (a 2:1 split between the two determiners). The paper uses each speaker's **per-noun** bias (share of tokens taking each noun's favored determiner, about .80–.85). With b = 2/3 the Providence children pass (p .60). With the per-noun bias they fail (p .004). No single bias makes every group pass: b = 2/3 passes Providence but fails the Manchester children and every group of mothers.

## 1. The extractions

Child sites, shared pipeline, five Providence children (Naima, Lily, Alex, William, Ethan); Lara alongside.

| Child | Ages (months) | Sites | Imitations* | Letter names | Hapax nouns |
|---|---|---|---|---|---|
| Naima | 13–46 | 5,447 | 12.6% | 17 | 354 / 971 |
| Lily | 15–48 | 3,192 | 13.1% | 19 | 321 / 838 |
| Alex | 18–42 | 1,803 | 10.2% | 25 | 154 / 385 |
| William | 16–41 | 1,325 | 12.2% | 25 | 137 / 360 |
| Ethan | 13–35 | 1,697 | 14.6% | 6 | 276 / 552 |
| Lara | 21–40 | 4,046 | 10.8% | 2 | 206 / 613 |

\* Same det + noun in an adult utterance in the previous 2 utterances.

- A hand check of 20 random sites on nouns used with both determiners found real child noun phrases ("that's a microphone", "spray the fire", "the kitty has the ball of yarn"). There are a few nursery rhymes ("a pocket full of posies", "like a diamond in the sky", "down by the bay").
- There are two oddities. Alex labels colors as nouns ("a red", "a pink"; 194 sites). Some sites are utterance-internal repeats ("the pig a pig a pig a pig"; 15–194 per child).
- **None of this drives the gap.** Removing imitations, letters, colors, "one" and within-utterance repeats *raises* the gap (+.054 to +.103 per child, p .002).
- **Tagger.** CLAN's own `%mor` tags give the same gap as spaCy on the same utterances: Providence children +.056 (p .022) with `%mor`, +.063 with spaCy, 5/5 positive both ways.

## 2. Age windows and samples

- **Ages.** Restricting to 24–36 months does not remove the gap (Naima +.058, Lily +.109, Ethan +.050). Before 30 months, Alex and William are close (+.015, +.001), but Naima, Lily and Ethan are not.
- **Samples.** In 6-month samples (≥100 sites), the Providence children over-overlap in **20 of 20** samples. The correlation between empirical and predicted is high (r .90), but the concordance is moderate (ρc .65). Manchester's 27 samples over-overlap in 18 of 27 (ρc .88, paired p .10).

## 3. Bias definition

Paired t-test of empirical vs predicted overlap under three bias definitions.

| Group | Empirical | Per-noun bias (paper) | Overall max(P(the), P(a)) | Fixed 2/3 (Yang 2013) |
|---|---|---|---|---|
| Providence children (5) | .303 | .240, p .004 | .329, p .11 | .310, p .60 |
| Providence mothers (5) | .359 | .366, p .57 | .474, p < .001 | .446, p .001 |
| Manchester children (12) | .259 | .251, p .66 | .352, p < .001 | .341, p < .001 |
| Manchester mothers (12) | .307 | .328, p .16 | .450, p < .001 | .423, p < .001 |
| All 24 children | .288 | .268, p .056 | .365, p < .001 | .348, p < .001 |

- **Where 2/3 comes from.** The six expected values in Yang's table (Naomi .218, Eve .254, Sarah .288, Adam .337, Peter .422, Nina .451, with his S and N) are each reproduced by b = .665–.669 in the paper's formula.
- **Per-noun bias across children.** Under the per-noun bias, the gap tracks tokens per noun type across the children (r −.64). Children with few tokens per type over-overlap (Anne, Becky, Lily, William), and children with many under-overlap (Carl, Warren, John). Most Providence children sit at the low end.
- **Providence against Manchester.** Even after allowing for tokens per type, Providence is about .05 above Manchester's fit (p .01). Manchester passes as a group because Carl, Warren, John and Aran pull the other way. Eight of the 12 Manchester children over-overlap too.

## What drives the gap

The Providence children's det–noun lists are clean, and the result holds with CLAN's own tags. They over-overlap relative to the paper's per-noun-bias prediction at every age. Charles's "pass" is what Yang's 2:1 bias gives for these five children. Under that bias the Manchester children and all mothers fail. So the two results disagree because the bias differs, not because of the data.

Still to confirm with Charles: which bias he used, his sample definition (age window, sessions), and his pass criterion. His own S, N and empirical value for one child (e.g. Lily) would settle it.

Scripts: `scripts/providence_audit/` on the baselines branch. CSVs: `results/baselines/all_children/providence_audit/`.

## 4. Literal CHILDES `%mor` extraction (Nov 24, follow-up)

Pairs taken straight from each utterance's `%mor` tier, with no cleaning and no spaCy. A pair is an article tag (`det:art|a/an/the`; in Manchester's 2024 UD-style `%mor`, `det|a-…-Art`) immediately followed by a common noun (`n|` / `noun|`), with the noun as its `%mor` stem. Three variants: any number; singular only; singular with one adjective allowed in between. `scripts/providence_audit/mor_literal_dxn.py` → `providence_audit/mor_literal_dxn.csv`.

**Children, per child (ours → literal `%mor`, singular).** Providence: Alex S 1,803 → 1,759, N 385 → 345, emp .309 → .336; Ethan 1,697 → 1,581, 552 → 509, .250 → .265; Lily 3,192 → 3,069, 838 → 733, .298 → .322; Naima 5,447 → 5,230, 971 → 879, .349 → .361; William 1,325 → 1,253, 360 → 322, .308 → .342. The `%mor` stems merge more forms ("doggie" → dog), so N drops by 8–13% and empirical overlap rises by .01–.03. The predicted overlap rises by about the same. Manchester changes by less than .03 in every cell.

| Group | Extraction | Per-noun bias | Fixed 2/3 |
|---|---|---|---|
| Providence children (5) | ours | .303 vs .240, p .004 | .303 vs .310, p .60 |
| | `%mor`, any number | .337 vs .270, p .016 | .337 vs .336, p .99 |
| | `%mor`, singular | .325 vs .262, p .008 | .325 vs .330, p .76 |
| | `%mor`, + adjective | .340 vs .278, p .016 | .340 vs .343, p .89 |
| Manchester children (12) | ours | p .54 | p < .001 |
| | `%mor` (all three) | p .62 / .84 / .79 | p ≤ .001 |
| Providence mothers (5) | `%mor` (all three) | p .35 / .23 / .43 | p < .001 |
| Manchester mothers (12) | `%mor` (all three) | p .93 / .95 / .10 | p < .001 |

**Verdict: extraction alone does not change pass/fail.** With the literal `%mor` pairs, Providence children still fail under the per-noun bias and pass under 2/3. Manchester children and every group of mothers pass under the per-noun bias and fail under 2/3. The bias decides the outcome. The extraction does not.

## 5. Charles Yang's code (Yang & Valian 2020 repo), Nov 25

Source: github.com/charles-yang-upenn/Determiner-Productivity (`Analysis 1 & 2/overlap.py`, `overlap_phase.py`). I re-implemented it line by line rather than running the downloaded code: `scripts/providence_audit/yang_valian_reimpl.py`. **Check:** on the repo's own Manchester files, the re-implementation reproduces all 24 rows of its Table 1 and all 12 rows of its Table 2 to the third decimal. Run on the Providence tpr-data files (same `<file> %mor:` line format).

**Differences from our pipeline**

| # | His code | Ours | Changes pass/fail? |
|---|---|---|---|
| 1 | Pairs from `%mor`: `det:art|a/an/the` + `n|X` with no "PL" in the tag; also article + any word + `n|Y` (adjective + noun, second noun of a compound) | spaCy on cleaned text: article + optional modifier + singular common noun | No |
| 2 | Noun = raw `%mor` string after `n|` (`dog-DIM`, `zip&dv-AGT` are their own types) | spaCy lemma | No |
| 3 | **Bias = per-noun dominant share** (`find_bias`), the same as ours. The formula is identical too. | same | — |
| 4 | Speakers: `*CHI` and `*MOT` only, all ages (Table 1) | child, mother; other adults only for TPR | No |
| 5 | **Speaker bug:** after a `*CHI:` line, the *next* `%mor` line is taken even when the child's line has none, so the following speaker's `%mor` counts as the child's. Manchester: 4% of child lines. Providence: 14–26% of child lines (child utterances without `%mor` are common there). | each `%mor` stays with its own utterance | No; it inflates child S (Lily 5,633 vs 3,592 corrected) |
| 6 | **Wrapped lines:** only the first line of a `%mor` tier is read. 25% of Providence `%mor` tiers wrap; his Manchester files are unwrapped. | continuation lines joined | Yes, for **mothers**: unjoined, Providence mothers over-overlap and fail (p .015); joined they pass (p .26) |
| 7 | Table 2 "phase one": child pairs up to 50 distinct det+noun types, plus the rest of that session. The session key is `filename[:-5]`. On tpr-data's prefixes (`010102`) that is `"0"`, so phase one silently becomes the whole corpus. | — | see below |
| 8 | No statistical test in the repo: tables of empirical vs expected only | paired t-test + JZS BF₀₁ | Possibly — see below |

**Providence children under his code** (paired t of empirical vs expected, 5 children):

| Variant | Empirical | Expected | p | Over-overlap |
|---|---|---|---|---|
| As written (unjoined lines, speaker bug) | .337 | .270 | .007 | 5/5 |
| Lines joined | .344 | .294 | .022 | 5/5 |
| Lines joined + speaker bug fixed | .326 | .271 | .023 | 5/5 |
| Phase one (session rule fixed) | .110 | .089 | .046 | 5/5 |
| Manchester children, his files, all ages | .295–.299 | .282–.285 | .47–.52 | 7/12 |
| Manchester children, his phase one (Table 2) | .118 | .118 | .99 | 8/12 |

**What this means.** With his own code and per-noun bias, the Providence children over-overlap and fail a paired t-test at every setting. In phase one they are close (.110 vs .089, differences of .002–.047 per child): that is the only setting where a judgment by eye could call them a match, and even there all 5 are above and p = .046. So Charles's "pass" is most likely the criterion (closeness of empirical and expected per child, without a test), or a phase-one style sample, not the bias or the extraction. The earlier 2/3 explanation does not apply to this code.

CSVs: `providence_audit/yang_valian_reimpl*.csv`, `yang_valian_phase1.csv`.
