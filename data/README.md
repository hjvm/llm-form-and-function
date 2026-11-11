# Data

The transcripts are not redistributed here. Steps 1 and 2 of the pipeline expect:

```
data/CHILDES/
├── Manchester/            12 child–mother dyads (Theakston et al. 2001)
│   ├── Anne/*.cha
│   ├── Aran/*.cha
│   └── ...                Becky, Carl, Dominic, Gail, Joel, John, Liz, Nicole, Ruth, Warren
└── tpr-data/              17 dyads with other adults present, one file per child
    ├── adam.cha
    └── ...
```

Steps 3–5 do not read `data/`.

## Manchester

Download the Manchester corpus from the CHILDES English-UK collection at TalkBank: <https://childes.talkbank.org/access/Eng-UK/Manchester.html>. TalkBank asks for a free account. Unpack it so that each child has a folder of `.cha` files named by age (`YYMMDD.cha`), e.g. `data/CHILDES/Manchester/Gail/020609.cha`.

Only the `*CHI:` and `*MOT:` tiers are used. CHAT annotations are stripped by `determiner.corpus.clean_childes_annotations`.

## tpr-data

The other-adult reference for the TPR test comes from 17 children's transcripts drawn from several CHILDES corpora (Brown, Manchester, Providence, Bloom, Suppes, Lara), preprocessed into a single file per child, `data/CHILDES/tpr-data/<child>.cha`. Each line is

```
<session id> <speaker tier> <utterance>
```

for example `adam01 *MOT: why don't you look at some of the toys in the basket.` Speakers other than `*CHI:` and `*MOT:` are treated as other adults.

## Do I need the data?

* **Paper tables and figures** (`python run_pipeline.py --only 5`): no.
* **Human baselines** (Step 2): yes, both corpora.
* **Model metrics** (Steps 3–4): the model predictions from Step 1, plus Step 2's TPR cases.

The determiner–noun sites used in the paper are already committed in `results/site_inventory/`, extracted from these transcripts with spaCy `en_core_web_sm` 3.7.1, so the transcripts are only needed for the conversation text around each site.
