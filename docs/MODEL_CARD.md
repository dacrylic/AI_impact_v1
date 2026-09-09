# GPT-5.2 Current-Guidance Model Card

## Purpose

This CPU-only classifier distils the approved current POC AI-impact scoring
policy into `E0`, `E1`, and `E23`. `E23` is the operational merge of the
source `E2` and `E3` labels. It is a proxy for the approved LLM policy, not
independently adjudicated ground truth about AI impact.

## Inputs

Primary API input:

- `action` and `object`, with optional `purpose`.

The API joins those supplied AOP fields into the same normalized atomic task
form used in training; it does not infer the fields from prose.

Compatibility and optional context:

- `keytaskContent`: an already-composed normalized task, accepted instead of
  action/object for upstreams that already provide it.
- `jobroleTitle`

The model never uses identifiers, sector, source labels, scores, bands, or
LLM rationale columns as inference features. When optional AOP fields are not
available, the API supplies blank values.

## Method

One CPU sparse `LinearSVC` consumes a concatenation of field-aware TF-IDF
blocks: task word n-grams, two task character n-gram views, title word n-grams,
individual action/object/purpose word n-grams, and an action-object lexical
cross. This preserves the practical benefits of the decomposed task format
without an embedding runtime, GPU requirement, or second model.

## Evaluation

The release training command exact-deduplicates complete model inputs before
splitting. It removes label-conflicting identical inputs rather than choosing
an arbitrary target, retains one deterministic representative of same-label
duplicates, then applies a role-grouped split.

| Check | Result |
| --- | ---: |
| Source rows | 77,469 |
| Full-input duplicates collapsed | 1,160 |
| Conflicting full inputs excluded | 0 |
| Training rows | 48,288 |
| Validation rows | 12,611 |
| Sealed test rows | 15,410 |
| Shared role IDs across splits | 0 |
| Shared complete inputs across splits | 0 |
| Sealed test accuracy | 0.8244 |
| Sealed test macro F1 | 0.8183 |
| Sealed test weighted F1 | 0.8236 |

Sealed-test class metrics:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.8614 | 0.8445 | 0.8528 | 4,166 |
| E1 | 0.7904 | 0.7316 | 0.7599 | 4,027 |
| E23 | 0.8212 | 0.8646 | 0.8423 | 7,217 |

The generated `training_report.json` contains the confusion matrix, source
data SHA-256, artifact SHA-256, split seed, and complete fixed feature recipe.
The binary artifact is intentionally not committed to Git; promote it through
the release process described in the deployment guide.

## Intended Use And Limits

Use this model for real-time task-level scoring with its review signal. Review
is especially appropriate for low-margin, low-lexical-coverage, or very short
tasks. The signal is a triage measure, not a calibrated probability of
correctness.

The held-out test measures generalization to unseen job roles in the approved
SFW-derived dataset. It does not establish performance on new sectors,
languages, task extraction errors, or semantic concepts absent from the source
corpus. Monitor review-score distributions and collect adjudicated production examples for
future retraining.
