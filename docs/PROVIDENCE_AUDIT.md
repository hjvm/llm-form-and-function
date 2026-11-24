# Providence children and the D×N test (Nov 24)

**Question.** Charles reports that the Providence children pass the D×N test with the Goldin-Meadow & Yang (2017) method. In our pipeline they over-overlap (empirical > predicted) and fail. Why?

**Short answer.** The det–noun lists are fine. The difference is the **bias** that goes into the predicted overlap. Yang's published expected values all correspond to a bias of about **2/3** (a 2:1 split between the two determiners). The paper uses each speaker's **per-noun** bias (share of tokens taking each noun's favored determiner, about .80–.85). With b = 2/3 the Providence children pass (p .60). With the per-noun bias they fail (p .004). No single bias makes every group pass: b = 2/3 passes Providence but fails the Manchester children and every group of mothers.

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
