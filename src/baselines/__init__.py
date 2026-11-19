"""Follow-up baselines for the D×N overlap and TPR tests (not part of the BabyLM 2026 paper).

Every baseline is scored exactly as the paper scores the language models: at the attested
Manchester D×N sites of ``results/site_inventory/manchester_det_noun_locations.csv``, with
analytical overlap / bias / TPR from P(the) at each site, the paired D×N t-test across the 12
dyads (child and caretaker sites separately) and the one-sample TPR test against the pooled
adult mean.

Modules
-------
common       shared loading, scoring and the closed-form overlap expectations
coins        coin-flip baselines (fair, 0.82, noun-level P(the | noun))
memory       Yang's (2013) memory model, by simulation and in closed form
ngram        5-gram baselines (Kneser-Ney on AO-CHILDES; Wikipedia KenLM scores)
zipf_audit   what the D×N test measures at fixed sites (Zipf, bias estimator, decomposition)
other_adults D×N pairs of the Manchester adults other than the mother (memory-model input)
figures      overlap scatterplots in the paper's style

Run everything with ``python pipeline/06_baselines.py``; see docs/BASELINES.md.
"""
