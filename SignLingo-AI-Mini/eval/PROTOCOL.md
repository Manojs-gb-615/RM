# ASL Signer Evaluation Protocol

**STATUS: PENDING HUMAN EVALUATION.** No rating in this repository is real yet.
Do not report ASL accuracy until this protocol has been completed by qualified signers.

## Goal

Measure whether retrieved pose clips are acceptable ASL for the transcript sentence,
and whether context-aware retrieval is better than the text-only baseline.

## Participants

- At least 3 Deaf ASL signers (ideally 5), fluent in American Sign Language.
- Raters must not have built the system. Blinded to retrieval mode where possible.

## Stimuli

1. Build the real index (`python ml/build_real_index.py ...`).
2. Sample 20 transcript sentences from the validation split, stratified by duration.
3. For each sentence, capture both `text_only` and `context_aware` retrieved clips.
4. Render each clip as an animation or video, labelled only with a random
   `sample_id` (e.g. `E01a`, `E01b`). Do not show the mode name.
5. Randomize presentation order per rater.

## Procedure

For each stimulus, the rater watches the signed clip alongside the English sentence
and scores:

| Dimension | Question | Scale |
|---|---|---|
| Sign accuracy | Does the signing convey the meaning of the English sentence? | 1 (wrong) – 5 (fully accurate) |
| Naturalness | Does the movement look like fluent, natural ASL? | 1 (unnatural) – 5 (natural) |
| Sync quality | Does the pose motion match the sentence timing / rhythm? | 1 (badly out of sync) – 5 (well synced) |

Free comments are encouraged (especially error types: wrong sign, missing
non-manual marker, handshape error, body orientation, etc.).

## Recording

Fill `eval/rating_form.csv` (one row per rater × stimulus × mode):

```
rater_id,sample_id,mode,sign_accuracy_1_5,naturalness_1_5,sync_quality_1_5,comments
```

Score values must be integers 1–5. Rows whose `sample_id` starts with `EXAMPLE`
are ignored.

## Aggregation

```bash
python eval/aggregate_ratings.py \
    --input eval/rating_form.csv \
    --output eval/summary.csv
```

The script reports mean, standard deviation and n per mode and metric, and prints
`PENDING HUMAN EVALUATION` while the form is empty. Report per-mode means with
confidence intervals and disclose rater count, blinding, and sampling.
