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

The 5-grams are run through the paper's own CAC pipeline as autoregressive models (next section). Their Step 1 predictions are committed in `results/baselines/ngram/<model>_cac_predictions.csv.gz`.

Outputs go to `results/baselines/` and `figures/baselines/`.

## The 5-grams go through the paper's pipeline

`src/cac/ngram.py` wraps each 5-gram as an `ar` model, and the unchanged Step 1 runs it:

```
python pipeline/01_run_inference.py --models configs/ngram_models.json
python pipeline/06_baselines.py --collect-ngrams --only ngram zipf figures
```

- **Prompts.** Step 1 builds the same discourse prompt as for GPT-2: the preceding session lines labelled `*CHI:`/`*MOT:`, then the target line with `___`, trimmed to the same 1,024-token budget. The target lines are identical to GPT-2's.
- **Scoring.** Each form (*a*, *an*, *the*) is put in the blank and the whole prompt is scored. A line break is an utterance boundary, since both models were trained one utterance per line, so the prompt's log-probability is the sum of its lines' sentence log-probabilities. The context lines are the same for every form and cancel in the normalisation.
- **Normalisation.** `merge_and_normalize`, as for every model: P(a) = logaddexp(a, an), then a softmax.
- **Metrics.** Step 3 unchanged (analytical overlap, bias, predicted overlap, TPR), then the D×N and TPR tests of notebooks 04 and 05.
- **Validation.** On GPT-2's paper predictions, the same Step 3 and tests give the committed values exactly: D×N t = 2.476, BF₀₁ = 0.414; TPR 0.252, t = 2.981, p = 0.012.

Models: `aochildes-kn5`, an interpolated Kneser-Ney 5-gram trained on AO-CHILDES (Huebner & Willits 2021, <https://github.com/UIUCLearningLanguageLab/AOCHILDES>; needs `data/AOCHILDES/aochildes.txt`); `wikipedia-kenlm5`, the public KenLM model `BramVanroy/kenlm_wikipedia_en` (needs `kenlm`, spaCy `en_core_web_sm` and `wiki_en_token.arpa.bin`).

An earlier version scored each target utterance on its own, outside the pipeline. Its results are withdrawn. Re-scoring the target line without its speaker label reproduces those old probabilities exactly, so the only difference is the `*CHI:`/`*MOT:` label, which a 5-gram can see when the determiner is near the start of the line.

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

Numbers are from `results/baselines/`. Accuracy is at child / caretaker sites, pooled over sites. D×N and TPR are at child sites, scored analytically; "draws" means the share of 1,000 realised transcripts that pass.

| Baseline | Accuracy | D×N: emp vs pred | D×N p (BF₀₁) | Draws passing D×N | TPR (vs 0.215) |
|---|---|---|---|---|---|
| Children | | 0.259 vs 0.251 | 0.659 (3.18) | | 0.226, passes |
| Fair coin | | 0.448 vs 0.367 | 0.002 | 0% | 0.500, fails |
| 0.82 coin | | 0.322 vs 0.259 | < 0.001 | 0% | 0.502, fails |
| Noun-level coin, P(the given noun) | | 0.321 vs 0.312 | 0.592 (3.05) | 46% | 0.346, fails |
| Memory model, all adults | | 0.218 vs 0.155 | < 0.001 | 0% | n/a |
| 5-gram, child-directed (AO-CHILDES), through CAC | 0.75 / 0.82 | 0.358 vs 0.320 | 0.049 (0.59) | 0% | 0.357, fails |
| 5-gram, Wikipedia, through CAC | to be run | | | | |

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

The 5-grams are scored on the child's nouns, so they cannot undershoot that way. At every noun frequency, the child-directed 5-gram puts both determiners on a noun more often than the children do (nouns used 3–4 times: 0.58 against 0.39; 5–9 times: 0.80 against 0.59; `overlap_by_noun_frequency.csv`). Its per-site probabilities hedge, and with four words of context it cannot see that a referent was introduced earlier in the session.

## Files

| File | Contents |
|---|---|
| `coin_results.csv` | Coins: analytical and sampled. |
| `memory_results.csv` | Memory model: closed form, draws, Yang's comparison, D×N. |
| `memory_per_dyad.csv` | Memory model, per dyad. |
| `ngram_results.csv` | The 5-grams, with accuracy. |
| `ngram/<model>_cac_predictions.csv.gz` | Step 1 predictions of each 5-gram: `sentence_id`, `speaker`, `P_a`, `P_the`, `truncated`. |
| `zipf_decomposition.csv` | emp, Y, C, `zipfian_fit` slope and frequency spectrum per speaker sample. |
| `zipf_noun_coins.csv` | Coins on Zipf-drawn nouns vs on transcript nouns. |
| `formula_self_check.csv` | Data from the formula's own assumptions, scored with b̂ and with the true b. |
| `session_consistency.csv` | Same-determiner rate within vs across sessions. |
| `overlap_by_noun_frequency.csv` | Overlap at child sites by noun frequency, per model. |
| `inputs/manchester_other_adult_det_noun.csv` | D×N pairs of the non-mother adults (spaCy 3.8 / en_core_web_sm 3.8.0). |
