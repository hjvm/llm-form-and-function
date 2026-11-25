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

## 5. Yang & Valian (2020) and its repository (corrected Nov 25)

Paper: Yang & Valian (2020), "Determining the abstractness of determiners" (manuscript; the user's Zotero copy, also at ling.upenn.edu/~ycharles/papers/YV2020.pdf). Repo: github.com/charles-yang-upenn/Determiner-Productivity. The repo bundles **five different analyses**; the first version of this section mixed them. Mapping:

| Paper | Question | Repo | Data / sample | Expected value | Criterion |
|---|---|---|---|---|---|
| Analysis 1 (Fig. 1) | Replicates Valian et al. 2009: does overlap rise with a noun's opportunity count? | `controlfreq.py`, `runcontrolfreq` | Manchester 12 children + mothers, full corpus | none | correlations; child vs mother paired t |
| **Analysis 2 (Tables 1–2)** | **Replicates Yang 2013: observed vs expected overlap** | **`overlap.py` (Table 1), `overlap_phase.py` (Table 2)** | Manchester 12 + mothers; Table 1 full corpus, all nouns; Table 2 "phase one" (child pairs until 50 unique Det-N pairs, plus the rest of that session) | Yang's Zipf formula with each **speaker's own per-noun bias** (dominant share summed over nouns ÷ tokens) | **paired t-test**, observed vs expected (children p .47, mothers p .25; phase one p .99) |
| Analysis 3 (Tables 3, 4, 6) | Replicates Pine et al. 2013 child vs mother comparison | `Analysis 3/pine.py` | Pine phases; nouns used with both a and the by the mothers; child ≥2 uses per phase; resampling, 100 simulations | none (empirical vs empirical) | paired t |
| Analysis 4 (Tables 5, 6) | Pine's method on adult vs adult (Mac: Mary vs Brian; Bloom: Peter's mother vs Patsy, Lois) and mother vs herself | `Analysis 4/`, `pine.py` | MacWhinney, Bloom | none | paired t |
| Analysis 5 (Tables 7–9) | Formal analysis of why Pine's sampling deflates the larger sample | `Analysis 5/pinebias-*.py`, `pineclump.py` | Manchester | o(b,f), O(B,F) | paired t |

**The analysis behind the Providence claim is Analysis 2** (Yang's overlap test, which is also what Goldin-Meadow & Yang 2017 apply). The `masterlist`, "≥2 uses" and resampling belong to Analyses 3–5 only and play no part here; neither run below uses them. Analysis 2 uses the **same bias definition and formula as our paper**, and judges the match with a **paired t-test**, as we do. (Goldin-Meadow & Yang 2017 instead report Lin's concordance across samples.) The fixed 2/3 in sections 3–4 reproduces Yang's 2013 table only; it is not what Analysis 2 or GM&Y use.

**Checks.** My line-by-line re-implementation of `overlap.py` and `overlap_phase.py` (I did not run the downloaded code) reproduces all 24 rows of Table 1 and all 12 rows of Table 2 on the repo's Manchester files.

**Extraction in Analysis 2** (paper p. 7 + code): `%mor` article (`det:art|a/an/the`) followed by a singular `n|` noun (tag without "PL"), also with one intervening word ("following Pine et al. 2013"; the paper says excluding these does not change the results); noun = the raw `%mor` string (`dog-DIM` separate from `dog`); `*CHI` and `*MOT` only. Our pipeline: spaCy on cleaned text, lemmas. Two code issues matter only on Providence-format files: (a) after a `*CHI:` line without `%mor`, the next speaker's `%mor` is counted as the child's (Manchester 4% of child lines, Providence 14–26%); (b) only the first line of a wrapped `%mor` tier is read (25% of Providence tiers wrap; Manchester repo files do not wrap). Unjoined, the Providence mothers fail (p .015); joined they pass (p .26). The paper says Table 2's phase one starts at 2;0; the code starts at each child's first session (this is what reproduces Table 2), and its session key `filename[:-5]` matches every session on tpr-data names.

**Providence under Analysis 2** (5 children; paired t, empirical vs expected):

| Sample | Variant | Empirical | Expected | p | Over |
|---|---|---|---|---|---|
| Table 1 (full corpus) | as written | .337 | .270 | .007 | 5/5 |
| | lines joined | .344 | .294 | .022 | 5/5 |
| | lines joined + speaker fix | .326 | .271 | .023 | 5/5 |
| Table 2 (phase one, S ≈ 80–150) | as written, from first session | .110 | .089 | .046 | 5/5 |
| | as written, from 2;0 | .122 | .093 | .024 | 5/5 |
| | speaker fix, from first session | .148 | .141 | .42 | 3/5 |
| | speaker fix, from 2;0 | .164 | .135 | .020 | 5/5 |
| Manchester (repo files) | Table 1 / Table 2 | .30 / .12 | .28 / .12 | .47 / .99 | 7/12, 8/12 |

**Conclusion.** Under the analysis that corresponds to the claim (Analysis 2, per-speaker bias, paired t), the Providence children fail on the full corpus in every variant (p .007–.023). Phase one is small and unstable: it fails in three variants and passes in one (p .42). A concordance criterion would be more lenient: across the 5 children ρc = .37, but pooled with their mothers ρc = .78, about the same as Manchester's children (.79), because the offset of a few points is small against the spread across speakers. If Charles's "pass" is a concordance judgment as in GM&Y 2017, or a phase-one sample, that would explain it; under Analysis 2's own paired t-test on the full corpus, the Providence children over-overlap.

Scripts: `scripts/providence_audit/yang_valian_reimpl.py`, `yv2020_analysis2_phase1.py`. CSVs: `providence_audit/yang_valian_reimpl*.csv`, `yv2020_analysis2_phase1.csv`.

## 6. Why the Providence children over-overlap (Nov 25)

Each speaker's gap (empirical − predicted, paper method) splits into three parts that add up exactly (`scripts/providence_audit/gap_decomposition.py`, 300 simulations per speaker):

- **Test offset**: the gap a perfectly productive speaker gets under Yang's own model (Zipf nouns, free a/the choice) with the same number of noun types, tokens and estimated bias. It comes from estimating the bias from the same small sample.
- **Noun-frequency shape**: how much more overlap the speaker's real noun frequencies give than Zipf frequencies with the same N and S, with a/the still chosen freely.
- **Determiner use**: the speaker's real overlap minus what the real noun frequencies would give with free a/the choice. Negative means nouns stick to one determiner more than chance.

Means per group (child rows; mothers in the CSV):

| Group | Gap | Test offset | Noun-frequency shape | Determiner use | Top-10 nouns' share of tokens (Zipf predicts) |
|---|---|---|---|---|---|
| Providence children (5) | **+.063** | +.025 | **+.101** | −.063 | .18 (.42) |
| Manchester children (12) | +.008 | +.024 | +.056 | −.073 | .24 (.46) |
| Yang's 6 children | +.015 | +.027 | +.063 | −.075 | .17 (.43) |
| Lara | +.004 | +.026 | +.065 | −.087 | .16 (.42) |
| Providence mothers (5) | −.008 | +.028 | +.048 | −.083 | .13 (.38) |
| Manchester mothers (12) | −.022 | +.026 | +.041 | −.089 | .18 (.40) |

Per Providence child (gap = offset + shape + determiner use): Alex +.042 = .026 + .081 − .066; Ethan +.057 = .021 + .093 − .059; Lily +.095 = .021 + .133 − .059; Naima +.040 = .030 + .089 − .079; William +.080 = .025 + .107 − .051. Manchester children who over-overlap look the same: Anne +.084 = .024 + .103 − .041; Becky +.068 = .023 + .093 − .050; Nicole +.058 = .019 + .075 − .036.

**Findings**

1. **The test offset is the same everywhere (+.02 to +.03)** and so does not separate Providence from Manchester. It is why even a perfectly productive speaker's empirical overlap lands slightly above the prediction (cf. the Spanish null simulations).
2. **Determiner use is similar** (Providence −.063 vs Manchester −.073). The Providence children do not spread a/the over nouns more freely than the Manchester children in any meaningful way.
3. **The difference is the noun-frequency shape** (+.101 vs +.056, i.e. +.045 of the +.055 difference in gap). Every child's noun distribution is flatter at the top than Zipf (the 10 most frequent nouns take 11–36% of tokens, while Zipf with the same N gives 39–52%). A flatter head means more nouns are used often enough to show up with both determiners than Yang's Zipf formula assumes. The flatter the top, the bigger the shape term (r = −.65 with the top-10 share across 24 children) and the bigger the gap (r = −.59). The Providence children have some of the flattest distributions (Lily's top 10 = 11% of tokens), and Manchester children with steep heads (Carl's "car", Ruth, Dominic, Warren) have small or negative shape terms that pull the Manchester mean back to zero.
4. **It is not sampling design.** Re-sampling each Providence child like Manchester (34 sessions from 2;0–3;0, 20 draws) leaves the gap at +.055 (shape +.089, top-10 share .21) and the five fail the paired t-test in 20 of 20 draws (median p .022). Number of sessions, S and N do not predict the shape term (|r| ≤ .32). It is not extraction either (section 4: the `%mor` pairs give the same gap).
5. **The mothers do not show it.** Providence mothers' shape term (+.048) matches Manchester mothers' (+.041).

**Plain-language cause.** Yang's prediction assumes noun frequencies follow Zipf's law with exponent 1, which concentrates tokens in a handful of very frequent nouns. Real child speech spreads tokens over more mid-frequency nouns, so it shows more overlap than predicted. In Manchester this is offset on average by children whose talk is dominated by a few nouns (Carl). The Providence children talk about a wider spread of things at middle frequencies, so the Zipf prediction undershoots them all, by about 4–10 points, and with no child pulling the other way the paired t-test fails. The overlap test, as formulated, is sensitive to how well each child's noun distribution matches Zipf, not only to productivity.

CSVs: `providence_audit/gap_decomposition.csv`, `gap_manchester_like_samples.csv`.
