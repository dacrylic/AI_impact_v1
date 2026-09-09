# Completed Factor Analysis: Legacy Reproduction, Production Policy, and Few-Shot Anchors

## What this analysis resolves

The original comparison was incomplete because the historical prompt had not
been reconstructed faithfully. The recovered colleague script closes that gap:
on the same 978 raw SFW parent tasks, the legacy prompt with GPT-4o produces
76.99% E0 versus 76.38% in the historical labels, with 95.30% exact four-class
agreement. It is therefore a credible operational reproduction of the historic
labeling policy for this panel.

This lets us distinguish a repeatable historical policy from later choices
rather than treating every observed difference as unexplained LLM drift.

## Study design

The completed dataset is a matched **3 prompt x 2 model x 2 task-grain** study.
Each of 978 simple-random raw parent tasks, and each of their 1,696 linked AOP
atomic children, was scored in every condition.

| Factor | Levels |
|---|---|
| Prompt policy | Legacy recovered script; updated production prompt; updated production prompt plus the legacy script's five few-shot examples |
| Model | GPT-4o (`gpt-4o-2024-08-06`); GPT-5.2 (`gpt-5.2-2025-12-11`) |
| Task grain | Original raw parent task; normalized/decomposed atomic child task |

There are 16,044 successful responses: 5,348 from the legacy reconstruction,
5,348 from the updated production run, and 5,348 from the few-shot-anchor run.
There are no duplicate call IDs within an individual run. The legacy rerun
reuses the original manifest call IDs, so apparent duplicates only occur when
the separate run files are concatenated; they are not duplicate API results.

The legacy-versus-production comparison is the clean 2 x 2 core. The anchored
prompt is a third intervention arm, not a second independent binary factor;
its purpose is to quantify whether the recovered examples can calibrate the
updated policy.

## Historical repeatability check

Historical labels are a reference for repeatability, not adjudicated ground
truth. They were never used to choose the representative panel or passed into
any scoring call.

| Raw-parent condition | E0 | Impacted (non-E0) | Exact agreement with historical label | E0 vs impacted agreement |
|---|---:|---:|---:|---:|
| Historical dataset | 76.38% | 23.62% | - | - |
| Legacy prompt + GPT-4o | 76.99% | 23.01% | **95.30%** | **96.11%** |
| Legacy prompt + GPT-5.2 | 38.55% | 61.45% | 57.98% | 62.17% |
| Updated production prompt + GPT-4o | 41.10% | 58.90% | 58.38% | 63.70% |
| Updated production prompt + GPT-5.2 | 31.90% | 68.10% | 50.61% | 55.11% |
| Updated prompt + legacy few-shot anchors + GPT-4o | 54.19% | 45.81% | 72.49% | 75.77% |
| Updated prompt + legacy few-shot anchors + GPT-5.2 | 35.79% | 64.21% | 55.32% | 59.20% |

The recovered legacy script is the only condition that recreates both the
historical distribution and individual historic labels. This strongly reduces
the likelihood that the historical 23.6% figure arose from a bad sample or a
systematic data leak. It is evidence of a different, reproducible labeling
policy.

## All condition distributions

`E23` is the governance merge of E2 and E3. “Impacted” means any non-E0 label.
All percentages are within the row's task grain.

| Prompt | Model | Grain | N | E0 | E1 | E23 | Impacted |
|---|---|---|---:|---:|---:|---:|---:|
| Legacy | GPT-4o | Raw parent | 978 | 76.99% | 12.58% | 10.43% | 23.01% |
| Legacy | GPT-4o | Atomic child | 1,696 | 74.53% | 15.86% | 9.61% | 25.47% |
| Legacy | GPT-5.2 | Raw parent | 978 | 38.55% | 21.68% | 39.78% | 61.45% |
| Legacy | GPT-5.2 | Atomic child | 1,696 | 37.21% | 25.24% | 37.56% | 62.79% |
| Updated production | GPT-4o | Raw parent | 978 | 41.10% | 13.80% | 45.09% | 58.90% |
| Updated production | GPT-4o | Atomic child | 1,696 | 41.75% | 15.27% | 42.98% | 58.25% |
| Updated production | GPT-5.2 | Raw parent | 978 | 31.90% | 29.24% | 38.85% | 68.10% |
| Updated production | GPT-5.2 | Atomic child | 1,696 | 32.43% | 32.25% | 35.32% | 67.57% |
| Updated + legacy few-shot anchors | GPT-4o | Raw parent | 978 | 54.19% | 16.67% | 29.14% | 45.81% |
| Updated + legacy few-shot anchors | GPT-4o | Atomic child | 1,696 | 53.12% | 18.63% | 28.24% | 46.88% |
| Updated + legacy few-shot anchors | GPT-5.2 | Raw parent | 978 | 35.79% | 31.08% | 33.13% | 64.21% |
| Updated + legacy few-shot anchors | GPT-5.2 | Atomic child | 1,696 | 37.38% | 32.13% | 30.48% | 62.62% |

## Factor effects on the matched raw-parent panel

The following are paired percentage-point changes; each 95% confidence interval
uses 2,000 non-parametric bootstrap replicates of the same 978 parents.

| Controlled change | E0 change | E1 change | E23 change | What it means |
|---|---:|---:|---:|---|
| Update prompt, holding GPT-4o | -35.89 [-38.85, -32.72] | +1.23 [-1.02, +3.58] | +34.66 [+31.69, +37.73] | Under GPT-4o, the updated policy shifts a large share of E0 directly into E23. |
| Update prompt, holding GPT-5.2 | -6.65 [-9.00, -4.29] | +7.57 [+5.21, +9.82] | -0.92 [-3.37, +1.64] | Under GPT-5.2, the update mostly moves E0 into E1 instead. |
| Change model to GPT-5.2, holding legacy prompt | -38.45 [-41.51, -35.38] | +9.10 [+6.54, +11.76] | +29.35 [+26.48, +32.21] | The legacy policy is highly model-sensitive. |
| Change model to GPT-5.2, holding updated prompt | -9.20 [-11.25, -7.16] | +15.44 [+13.09, +17.69] | -6.24 [-9.10, -3.48] | The updated policy narrows the model gap, but changes the E1/E23 mix. |
| Add legacy anchors, holding updated GPT-4o | +13.09 [+10.84, +15.34] | +2.86 [+1.33, +4.50] | -15.95 [-18.30, -13.50] | The five examples materially pull GPT-4o back toward the historical distribution. |
| Add legacy anchors, holding updated GPT-5.2 | +3.89 [+2.35, +5.42] | +1.84 [-0.20, +3.89] | -5.73 [-7.98, -3.58] | Anchors have a smaller but real calibration effect on GPT-5.2. |

### The important interaction

The model effect under the legacy prompt is **-38.45 points of E0** for
GPT-5.2 relative to GPT-4o. Under the updated prompt it is only **-9.20
points**. The 29.24-point difference is the prompt-by-model interaction.

Therefore, “GPT-5.2 doubles AI impact” is not a correct conclusion. GPT-5.2
is much more liberal under the recovered legacy policy; the updated policy
constrains that response differently and changes which non-E0 class absorbs
the shift. The observed outcome comes from the combination of model and policy,
not from a single independent cause.

## Does normalization/decomposition materially change the scoring distribution?

Raw parents and atomic children are not interchangeable observations: one
parent may contain several children. To avoid giving decomposed parents extra
weight, we compare each raw parent label with that parent's *mean child
outcome rate*, then bootstrap across parents.

| Condition | Raw E0 | Parent-weighted child E0 | Child minus raw E0 | Interpretation |
|---|---:|---:|---:|---|
| Legacy + GPT-4o | 76.99% | 73.93% | -3.07 pp [-4.65, -1.54] | Slight shift from E0 to E1 after decomposition. |
| Legacy + GPT-5.2 | 38.55% | 35.92% | -2.62 pp [-4.28, -1.00] | Small E0 reduction; E23 is statistically unchanged. |
| Updated + GPT-4o | 41.10% | 40.57% | -0.53 pp [-2.04, +0.89] | No material E0 difference. |
| Updated + GPT-5.2 | 31.90% | 30.83% | -1.07 pp [-2.65, +0.43] | No material E0 difference. |
| Updated + anchors + GPT-4o | 54.19% | 52.30% | -1.90 pp [-3.41, -0.38] | Small E0 reduction. |
| Updated + anchors + GPT-5.2 | 35.79% | 35.89% | +0.10 pp [-1.37, +1.55] | No material difference. |

This supports the operational choice of normalized AOP tasks: it gives the
score a clearer, more stable unit of meaning, while its effect on E0 prevalence
is small relative to the 9-38 point prompt/model effects. It does **not** prove
that raw and atomic labels should be aggregated into one national statistic;
they represent different units of work.

## What the five-shot arm tells us

The legacy examples are powerful calibration anchors, especially for GPT-4o.
They do not recreate the legacy condition because the updated production rubric
is still active. They should not be selected merely because they produce a
more politically comfortable midpoint between 23.6% and 68.1%.

Use them only if stakeholders deliberately choose a hybrid scoring policy and
document that policy. Otherwise they are best treated as evidence that examples
are an explicit policy lever, not as a default production addition.

## Decision implications

1. Preserve the historical dashboard statistic as a versioned legacy result.
   The recovered legacy GPT-4o recipe is now a strong audit/rollback reference.
   There is no evidence from this study that its 23.6% result was caused by the
   representative-sample design.
2. Treat the updated GPT-5.2 + normalized AOP policy as a new measurement
   policy, not a revision that can silently overwrite the dashboard. On this
   panel it produces 67.57% impacted atomic tasks, with the majority of the
   increase already visible at raw-parent grain (68.10%).
3. Do not choose the few-shot anchors as a compromise without policy approval.
   They are a real, measured third labeling policy.
4. Once management selects the desired LLM policy, train the low-latency ML
   classifier only on labels generated by that one policy. Mixing historical
   and current-policy labels would make the distilled target internally
   inconsistent.
5. For a product rollout, retain the legacy scorer in shadow/audit mode and
   monitor the operational model's E0 prevalence, OOV/review flags, and cohort
   drift. For public reporting, require separate governance approval and an
   explicit bridge narrative before changing the published national metric.

## Audit artifacts

The completed, machine-readable outputs are in
`results/factorial_completed_v2/`:

- `distributions.csv`: all 12 condition distributions and bootstrap intervals.
- `paired_prompt_and_model_effects.csv`: every matched prompt and model contrast.
- `parent_clustered_representation_effects.csv`: raw-versus-atomic comparisons
  with equal parent weighting.
- `historical_repeatability.csv` and `historical_confusion_matrices.csv`:
  the historical reproduction evidence.
- `raw_complete_cell_labels.csv` and `atomic_complete_cell_labels.csv`: every
  matched label used in the analysis.

The analysis is reproducible with
`src/ai_impact_classifier/experiments/analyze_completed_factorial.py`.
