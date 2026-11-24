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
