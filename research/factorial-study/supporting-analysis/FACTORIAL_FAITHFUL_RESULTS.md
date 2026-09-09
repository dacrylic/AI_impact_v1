# Faithful Appendix A.1 Factorial Results

These results supersede the earlier factorial outputs that used a shortened
paper-prompt reconstruction.

## Run integrity

- 10,696 successful calls: 978 raw parents and 1,696 linked atomic children
  across four prompt/model cells.
- No duplicate call IDs; all expected raw parents and atomic children are
  complete.
- Paper prompt: faithful transcription of the published Appendix A.1 rubric.
- Dynamic user content: occupation plus the relevant raw or atomic task only.
- Resolved model snapshots: `gpt-4o-2024-08-06` and `gpt-5.2-2025-12-11`.
- Production-prompt cells were retained from the prior complete run because
  their prompt hash matched the production prompt files.

## Raw-parent results

| Cell | E0 | Impacted (E1/E2/E3) |
|---|---:|---:|
| Historical labels | 76.4% | 23.6% |
| Faithful Appendix A.1 + GPT-4o | 40.4% | 59.6% |
| Faithful Appendix A.1 + GPT-5.2 | 8.3% | 91.7% |
| Updated production prompt + GPT-4o | 41.1% | 58.9% |
| Updated production prompt + GPT-5.2 | 31.9% | 68.1% |

The original-condition diagnostic is therefore the faithful Appendix A.1
prompt plus the raw parent task and GPT-4o. It does **not** reproduce the
historical 23.6% impacted result: it produces 59.6% impacted on the same 978
parents. Exact three-class agreement with the historical labels is 59.0%.

The earlier 47.2% impacted result was produced by a shortened prompt and must
not be used as the paper-prompt comparison.

## Raw-parent factorial effects

| Outcome | Prompt effect at GPT-4o | Prompt effect at GPT-5.2 | Model effect under paper prompt | Model effect under production prompt |
|---|---:|---:|---:|---:|
| Impacted / non-E0 | -0.7 pp | -23.6 pp | +32.1 pp | +9.2 pp |
| E1 | +2.1 pp | -45.7 pp | +63.3 pp | +15.4 pp |
| E23 | -2.9 pp | +22.1 pp | -31.2 pp | -6.2 pp |

Positive prompt effects mean the updated production prompt increases the
outcome relative to the faithful paper prompt. For non-E0, this means the
updated prompt is almost unchanged at GPT-4o but reduces impacted prevalence
by 23.6 percentage points at GPT-5.2.

## Interpretation boundary

This is a controlled sensitivity study, not proof of the historical dashboard
generation process. The paper describes an early GPT-4 system and says that
the GPT-4 prompt was modified to improve agreement with author labels; it does
not provide the historical model snapshot or API configuration. The faithful
replay shows that the published rubric itself is highly sensitive to model
generation, but cannot identify which undocumented choices produced the old
23.6% statistic.

Machine-readable outputs are in
`results/factorial_representative_v1/final_analysis_faithful/`. The combined
responses are in
`results/factorial_representative_v1/responses_faithful_factorial.csv`.
