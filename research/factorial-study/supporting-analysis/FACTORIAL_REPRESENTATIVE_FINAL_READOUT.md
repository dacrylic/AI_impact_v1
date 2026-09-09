# Factorial LLM-Label Drift: Final Representative Readout

## Executive summary

This study was designed to support a production decision: which scoring policy
should a real-time AI-impact API reproduce? It separates four possible sources
of LLM-label change:

1. the reconstructed legacy rubric versus the updated operational rubric;
2. GPT-4o versus GPT-5.2;
3. raw parent task text versus normalized atomic task text; and
4. their interactions.

The final run contains **978 simple-random matched raw parents** and their
**1,696 atomic children**. Every item was scored in all four prompt/model
cells, producing **10,696 successful Responses API calls**, with no errors or
duplicate call IDs. The simple-random sample was selected without using old
labels, new labels, or scores, so its marginal shares can be interpreted as
prevalence estimates for this panel.

The main result is that label drift is real and is not attributable to one
factor alone. GPT-5.2 produces substantially fewer E0 labels under the
legacy-rubric reconstruction, while the updated operational rubric moderates
that shift. The rubric/LLM interaction is therefore material. Decomposing
the same parents into atomic tasks changes prevalence only modestly relative
to the much larger prompt/model shifts, but it materially improves the unit of
analysis: one score now refers to one action-object-purpose unit rather than a
potentially multi-part parent task.

This distinction matters because the old study was a one-off analytical
exercise, whereas the intended system is a versioned real-time product. The
updated design reflects three advances: a later-generation LLM labeler, a more
explicit operational rubric, and AOP decomposition. It still follows the
same loose OpenAI/time-saving concept, including the question of whether an
LLM can reduce task time by at least half.

## Design and controls

- **Unit of sampling:** raw parent task, joined to its normalized atomic
  children through the retained linkage.
- **Sampling:** simple random selection of 978 matched parents (`seed=20260829`).
  This preserves an interpretable prevalence panel. A separate earlier
  label-stratified contrast panel is not pooled into these estimates.
- **Cells:** legacy-rubric reconstruction + GPT-4o; legacy-rubric
  reconstruction + GPT-5.2; updated operational rubric + GPT-4o; updated
  operational rubric + GPT-5.2.
- **Grains:** raw parent and atomic child are analyzed separately. Children
  are not treated as independent raw parents.
- **Target reporting:** E0, E1, E2, E3, and the governance merge E23 = E2 or
  E3. Distribution reporting is anchored on E0 prevalence.
- **Uncertainty:** 2,000 bootstrap replicates for E0 prevalence intervals.
- **Historical comparison:** old `openai_label` is used only as a reference
  for repeatability, never as an outbound API input or a sampling stratum.

The legacy condition is a documented reconstruction from the published rubric;
it is not represented as the exact historical tuned prompt.

## Raw-parent prevalence

| Condition | E0 prevalence (95% CI) | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| Legacy rubric + GPT-4o | 52.8% (49.6-55.9%) | 7.6% | 39.2% | 0.5% |
| Legacy rubric + GPT-5.2 | 22.9% (20.3-25.6%) | 60.5% | 15.8% | 0.7% |
| Updated rubric + GPT-4o | 41.1% (38.1-44.3%) | 13.8% | 44.5% | 0.6% |
| Updated rubric + GPT-5.2 | 31.9% (29.2-34.9%) | 29.2% | 38.4% | 0.4% |

The selected parents' historical E0 prevalence is 76.4%.
This is a descriptive reference, not a correctness benchmark for the new
LLM-generated labels.

## Atomic-child prevalence

| Condition | E0 prevalence (95% CI) | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| Legacy rubric + GPT-4o | 51.9% (49.6-54.2%) | 8.6% | 39.0% | 0.5% |
| Legacy rubric + GPT-5.2 | 25.9% (23.8-28.1%) | 58.8% | 14.9% | 0.4% |
| Updated rubric + GPT-4o | 41.7% (39.3-44.0%) | 15.3% | 42.5% | 0.5% |
| Updated rubric + GPT-5.2 | 32.4% (30.3-34.7%) | 32.3% | 34.9% | 0.4% |

Atomic decomposition slightly increases E1 under GPT-5.2 and slightly raises
E0 prevalence in the two GPT-5.2 cells. It does not explain the dominant
change from the legacy-rubric+GPT-4o cell to the legacy-rubric+GPT-5.2 cell.

## Main effects and interactions

Effects below are percentage-point contrasts on the paired panels. Positive
values mean the outcome becomes more common.

| Grain | Outcome | Rubric effect at 4o | Rubric effect at 5.2 | Model effect under legacy rubric | Model effect under updated rubric | Interaction |
|---|---|---:|---:|---:|---:|---:|
| Raw parent | E0 prevalence | -11.7 | +9.0 | -29.9 | -9.2 | +20.7 |
| Raw parent | E1 | +6.2 | -31.3 | +53.0 | +15.4 | -37.5 |
| Raw parent | E23 | +5.4 | +22.3 | -23.1 | -6.2 | +16.9 |
| Atomic child | E0 prevalence | -10.1 | +6.5 | -25.9 | -9.3 | +16.6 |
| Atomic child | E1 | +6.7 | -26.5 | +50.2 | +17.0 | -33.2 |
| Atomic child | E23 | +3.5 | +20.0 | -24.2 | -7.7 | +16.6 |

The most consequential interaction is for E1: the GPT-5.2 increase is very
large under the legacy-rubric reconstruction but much smaller under the
updated rubric. This is evidence against describing the change as simply
“GPT-5.2 is more liberal.” The updated rubric materially changes how the LLM
expresses its judgement.

## Historical repeatability

On the same raw-parent panel, comparing each new LLM-label condition with the
old `openai_label` after merging E2/E3 into E23 gives:

| Cell | Agreement with old labels | Predicted E0 prevalence |
|---|---:|---:|
| Legacy rubric + GPT-4o | 67.5% | 52.8% |
| Legacy rubric + GPT-5.2 | 41.6% | 22.9% |
| Updated rubric + GPT-4o | 58.5% | 41.1% |
| Updated rubric + GPT-5.2 | 50.7% | 31.9% |

The legacy-rubric+4o result is the closest available repeatability check, but
it is not perfect and the prompt is reconstructed. The large difference from
the old 76.4% E0 reference should therefore be presented as a combined
effect of prompt reconstruction, model version, and task representation, not
as a pure model-version effect.

In the stakeholder-facing formulation, the published panel reported **23.6%
AI-impacted tasks**. The closest controlled rerun reports **47.2%**, an increase
of **23.6 percentage points** and almost exactly 2x. The updated rubric with
GPT-4o raises the same share to 58.9%, while the current updated-rubric +
GPT-5.2 + atomic condition reaches 67.6%. These are methodology-version
differences, not a defensible time trend.

## Implications for the two production classifiers

The two trained classifiers are policy/data variants, not merely architecture
variants:

1. **Legacy classifier:** raw SFW task text trained on the historical
   GPT-4o/paper-derived labels. Its strongest strict duplicate-free task-only
   CPU LinearSVC benchmark reached macro F1 **0.7826** in five-fold CV.
2. **Current classifier:** normalized AOP task representation trained on the
   GPT-5.2/updated-rubric labels. The validation-selected field-aware TF-IDF +
   LinearSVC reached macro F1 **0.8188** on a fresh sealed test with zero
   shared roles and zero shared full input strings.

These scores are not an apples-to-apples leaderboard because the target
labels, task representation, permitted fields and evaluation design differ.
They show that each policy can be approximated by a low-latency CPU model.
Within the current-target experiments, a frozen MPNet blend reached **0.8250**
on the sealed test, but required roughly an order of magnitude more inference
time. The sparse LinearSVC therefore remains the preferred production
trade-off.

The principal data-science recommendation is deliberately two-track:

1. **Product:** approve the current classifier for a guarded internal pilot.
   Run both classifiers in shadow, monitor E0 prevalence, review rates,
   OOV/coverage signals and cohort drift, and route high-review cases for human
   inspection.
2. **National statistic:** do not immediately replace the published 23.6%
   estimate. Preserve it as a versioned historical result, build a formal
   bridge from the published labels through the closest rerun and each
   intended methodology change, and obtain stakeholder governance approval
   before any external restatement.

The legacy classifier remains useful for audit, comparison and rollback, not
as a competing production target. Mixing the two LLM labeling policies into
one target would make the classifier incoherent.

## Limitations and decisions still required

- The legacy prompt is reconstructed, so exact historical prompt repeatability
  cannot be claimed.
- The representative sample estimates this matched panel, not every possible
  job-posting distribution.
- Old labels are a historical reference, not adjudicated ground truth.
- E3 is rare in this panel; conclusions about E3 specifically are low-power.
- Atomic children are linked and clustered within parents; their percentages
  should not be read as an independent-population estimate without accounting
  for that clustering.

## Reproduction

```bash
./.venv/bin/python src/ai_impact_classifier/experiments/analyze_factorial_representative.py \
  --manifest results/factorial_representative_v1/panel/request_manifest.csv \
  --responses results/factorial_representative_v1/responses.csv \
  --parent-panel results/factorial_representative_v1/panel/parent_analysis_panel.csv \
  --child-panel results/factorial_representative_v1/panel/atomic_child_analysis_panel.csv \
  --output-dir results/factorial_representative_v1/final_analysis \
  --bootstrap-reps 2000
```

The machine-readable outputs are in
`results/factorial_representative_v1/final_analysis/`, including
`raw_distributions.csv`, `atomic_distributions.csv`, `paired_effects.csv`,
`factorial_effects.csv`, `historical_agreement.csv`, and the complete paired
label tables.
