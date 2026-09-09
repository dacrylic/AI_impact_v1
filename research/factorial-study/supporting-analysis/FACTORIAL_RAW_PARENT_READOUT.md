# Factorial Readout: Representative Raw Parents

## Status

This is the first frozen readout from the representative simple-random panel.
All 978 selected matched raw-parent records have four successful teacher
responses each. The atomic-child phase is still running and is not included
in the conclusions below.

The panel was sampled without using old labels, new labels, scores, or role
caps. Therefore its marginal label shares are suitable for prevalence
comparisons. The earlier label-stratified panel remains a separate contrast
sample and must not be pooled with this panel for prevalence estimates.

## Raw-parent results

| Condition | E0 | E1 | E2 | E3 | Non-E0 |
|---|---:|---:|---:|---:|---:|
| Paper prompt + GPT-4o | 52.8% | 7.6% | 39.2% | 0.5% | 47.2% (95% CI 44.2-50.3%) |
| Paper prompt + GPT-5.2 | 22.9% | 60.5% | 15.8% | 0.7% | 77.1% (95% CI 74.5-79.8%) |
| Production prompt + GPT-4o | 41.1% | 13.8% | 44.5% | 0.6% | 58.9% (95% CI 55.8-61.9%) |
| Production prompt + GPT-5.2 | 31.9% | 29.2% | 38.4% | 0.4% | 68.1% (95% CI 65.1-70.8%) |

The historical old-label distribution in the selected parents is retained in
the panel files for comparison, but it is not treated as ground truth for the
new teacher outputs.

## Interpretation

At raw-parent grain, both prompt and model contribute to label drift. The
largest visible shift is GPT-4o to GPT-5.2 under the reconstructed paper
prompt: non-E0 rises by 29.9 percentage points and E1 rises by 53.0 points.
Under the production prompt, the corresponding rises are 9.2 and 15.4
points. This indicates a prompt-by-model interaction rather than a simple
uniform model shift.

These results do not yet answer whether decomposition itself changes the
distribution. That requires the complete atomic-child cells, which are being
collected from the same matched parents. The paper condition is a documented
reconstruction from the published rubric, not a claim to possess the exact
historic tuned prompt.

## Reproduction

```bash
./.venv/bin/python src/ai_impact_classifier/experiments/analyze_factorial_representative.py \
  --manifest results/factorial_representative_v1/panel/request_manifest.csv \
  --responses results/factorial_representative_v1/responses.csv \
  --parent-panel results/factorial_representative_v1/panel/parent_analysis_panel.csv \
  --child-panel results/factorial_representative_v1/panel/atomic_child_analysis_panel.csv \
  --output-dir results/factorial_representative_v1/raw_parent_analysis \
  --bootstrap-reps 2000
```
