# Follow-up baselines for the overlap and TPR tests

This is follow-up work, **not part of the BabyLM 2026 paper**. It asks what the two tests do with
learners that we know lack grammar or discourse knowledge: coins, a memory model and 5-grams.

```
python pipeline/06_baselines.py                 # everything, 1,000 simulated transcripts per baseline
python pipeline/06_baselines.py --only coins    # one part: coins | memory | ngram | zipf | figures
```

Inputs:

- the committed site inventory;
- Step 2's TPR cases (`output/manchester_tpr_childes/tpr_cached_inputs.csv`);
- the committed human references in `results/tpr/`;
- `results/baselines/inputs/manchester_other_adult_det_noun.csv`, the non-mother adults' D×N pairs, extracted by `baselines.other_adults`.

The n-gram site scores are committed in `results/baselines/ngram/`. Re-scoring them (`--rescore-ngrams`) needs `data/AOCHILDES/aochildes.txt`. AO-CHILDES is from Huebner & Willits (2021), <https://github.com/UIUCLearningLanguageLab/AOCHILDES>. Re-scoring the Wikipedia model also needs `kenlm` and `wiki_en_token.arpa.bin` from `BramVanroy/kenlm_wikipedia_en`.

Outputs go to `results/baselines/` and `figures/baselines/`.

## Scoring

Every baseline assigns P(the) to each of the 75,133 attested sites, and P(a) = 1 − P(the). It is then scored exactly as the paper scores a language model:

- **Overlap and bias:** analytical, computed from the probabilities.
- **D×N test:** a paired t-test of empirical against predicted overlap across the 12 dyads, run separately for child and caretaker sites, with the JZS BF₀₁.
- **TPR test:** a one-sample t-test of the 12 per-dyad TPRs against the pooled adult mean, 0.215.

In addition, 1,000 transcripts are realised by drawing a determiner at every site. The tables report pass rates over these draws (p > 0.05).

## One formula, four special cases

Notation: noun n is used f_n times, and t_n and a_n = 1 − t_n are the probabilities of *the* and *a* at its sites.

**Nouns given.** With independent choices at each site, the noun shows both determiners with probability 1 − t_n^f_n − a_n^f_n. If the probabilities vary from site to site, this becomes 1 − ∏t − ∏a. The mean of this over nouns is the expected overlap, **C**. It is Charles Yang's "1 − the^s − a^s". With one t for every noun, it is `simple_expected_overlap()` in the original overlap code.

**Nouns drawn.** Here the nouns are drawn S times with replacement from π, so f_n ~ Binomial(S, π_n). Each noun then contributes

  E_n = 1 − (1 − π_n a_n)^S − (1 − π_n t_n)^S + (1 − π_n)^S

The expected overlap is ΣE_n over the expected number of distinct nouns drawn, Σ(1 − (1 − π_n)^S). These closed forms are implemented as `baselines.common.conditional_overlap` (C) and `unconditional_overlap` (E_n).

| Model | Nouns | P(the) |
|---|---|---|
| Paper's prediction **Y** (Yang 2013) | Zipf over N ranks | favoured determiner with probability b, the same for every noun |
| Memory model (Yang 2013) | drawn from the adult input's noun frequencies | input P(the given noun) |
| Constant coin | the child's nouns, fixed | q at every site, so the overlap equals C(q) |
| Noun-level coin, n-grams, every LM in the paper | the child's nouns, fixed | the model's P(the) at each site |

The memory model's closed form agrees with the mean of its 1,000 simulated draws to within 0.001 at every dyad. The draws are a Monte Carlo estimate of the formula.

## Results

Numbers are from `results/baselines/`. Accuracy is at child / caretaker sites. D×N and TPR are at child sites, scored analytically; "draws" means the share of 1,000 realised transcripts that pass.

| Baseline | Accuracy | D×N: emp vs pred | D×N p (BF₀₁) | Draws passing D×N | TPR (vs 0.215) |
|---|---|---|---|---|---|
| Children | | 0.259 vs 0.251 | 0.659 (3.18) | | 0.226, passes |
| Fair coin | | 0.448 vs 0.367 | 0.002 | 0% | 0.500, fails |
| 0.82 coin | | 0.322 vs 0.259 | < 0.001 | 0% | 0.502, fails |
| Noun-level coin, P(the given noun) | | 0.321 vs 0.312 | 0.592 (3.05) | 46% | 0.346, fails |
| Memory model, all adults | | 0.218 vs 0.155 | < 0.001 | 0% | n/a |
| 5-gram, child-directed (AO-CHILDES) | 0.77 / 0.83 | 0.348 vs 0.311 | 0.054 (0.64) | 0% | 0.341, fails |
| 5-gram, Wikipedia | 0.68 / 0.75 | 0.388 vs 0.333 | 0.008 | 0% | 0.399, fails |

The memory model is scored on its own sampled pairs. Yang's comparison of it with the children's empirical overlap gives 0.218 against 0.259 (Wilcoxon p = 0.001).

## What the D×N test measures when the nouns are fixed

Every model in the paper, and every baseline here except the memory model, is scored at the children's own sites. The nouns and their counts are therefore the children's, and only the determiners differ.

1. **A constant coin is the independence model itself.** Its analytical empirical overlap is C(q) and its bias is q, both exactly. Its D×N statistic is C(q) − Y(N, S, q), a function of the child's noun counts and q alone. The test learns nothing about the coin. The coin "fails" whatever happens, and always with *too much* overlap.
2. **The nouns are Zipf-like by slope, but not in the part of the spectrum that makes overlap.**
   - `zipfian_fit()` gives slopes of −0.93 for children and −1.10 for caretakers (R² 0.93). An exact Zipf(1) sample with the same S and the same number of types gives −0.90 and −1.00.
   - Against such a sample, though, the transcripts have more nouns used five or more times: 24% against 16% of types for children, 31% against 22% for caretakers. Children also have fewer hapaxes (43% against 50%).
   - Under a noun-blind determiner, mid-frequency nouns are where overlap comes from. On the real nouns the 0.82 coin gives 0.322; on Zipf-drawn nouns it gives 0.258.
3. **The bias is estimated from small counts.** b̂ = Σ max(#the, #a) / S. A noun seen once always counts as fully biased.
   - Take data generated exactly as the formula assumes (Zipf nouns, one b for every noun) and score them with b̂. At child-sized samples, overlap is 0.250 against a prediction of 0.229, and the 12-dyad test rejects in 98.5% of samples.
   - With the true b, the prediction is 0.251, and the test rejects in 2% of samples (`formula_self_check.csv`).
4. **Humans are below C.**
   - Given their own nouns, children's overlap is 0.259 against C = 0.305 with their own b̂ (p = 0.003). Caretakers' is 0.307 against 0.379 (p < 0.001).
   - Determiner use is noun-specific beyond a single b. Overlap is concave in a noun's lopsidedness, so unequal lopsidedness across nouns lowers overlap.
   - Determiner use is also consistent within a session. Two uses of the same noun take the same determiner 82% of the time within a session and 65% across sessions for children (p = 6×10⁻¹⁰). For caretakers it is 75% against 69%.
   - The exact decomposition emp − Y = (emp − C) + (C − Y) gives −0.046 + 0.054 for children and −0.072 + 0.051 for caretakers.

**At fixed sites, then, the D×N test compares a model's determiner choices with the children's, using Y as the yardstick.**

- A point above the line means the model's determiner choice is less committed to each noun than the children's is.
- A point below the line means it is more locked in than theirs.
- Noun-specific lexical statistics are enough to land on the children's spot: the noun-level coin passes. This is why the TPR test is needed.

## Why the n-grams overshoot while the memory model undershoots

The memory model draws its own nouns. With the child's S, a sample from the input contains about 530 noun types against the child's 339, so each noun is used fewer times. Keep the child's nouns and counts and use the input's P(the given noun) instead (C with t = input statistics). The memory model then gives 0.313, *above* the children's 0.259 (p = 0.0001).

The n-grams are scored on the child's nouns, so they cannot undershoot that way. At every noun frequency, they put both determiners on a noun more often than the children do: see `overlap_by_noun_frequency.csv`. Their per-site probabilities hedge, and a 5-gram cannot see that a referent was introduced earlier in the session.

## Files

| File | Contents |
|---|---|
| `coin_results.csv` | Coins: analytical and sampled. |
| `memory_results.csv` | Memory model: closed form, draws, Yang's comparison, D×N. |
| `memory_per_dyad.csv` | Memory model, per dyad. |
| `ngram_results.csv` | Both 5-grams, with accuracy. |
| `ngram/*_site_scores.csv.gz` | log10 P of each site's utterance with *a* / *the* / *an*. `row` is `sentence_id`. |
| `zipf_decomposition.csv` | emp, Y, C, `zipfian_fit` slope and frequency spectrum per speaker sample. |
| `zipf_noun_coins.csv` | Coins on Zipf-drawn nouns vs on transcript nouns. |
| `formula_self_check.csv` | Data from the formula's own assumptions, scored with b̂ and with the true b. |
| `session_consistency.csv` | Same-determiner rate within vs across sessions. |
| `overlap_by_noun_frequency.csv` | Overlap at child sites by noun frequency, per model. |
| `inputs/manchester_other_adult_det_noun.csv` | D×N pairs of the non-mother adults (spaCy 3.8 / en_core_web_sm 3.8.0). |
