# Production Output Schema

The production classifier is a single TF-IDF plus `LinearSVC` model. It always returns a class, but emits a review flag when the model is uncertain or the input activates unusually little of the trained vocabulary.

```json
{
  "schema_version": "1.1",
  "predicted_label": "E23",
  "needs_review": true,
  "review_level": "high",
  "review_score": 0.91,
  "review_reasons": ["low_decision_margin", "low_word_coverage"],
  "decision_margin": 0.18,
  "class_decision_scores": {"E0": -0.77, "E1": 0.06, "E23": 0.24},
  "word_active_features": 3,
  "char_active_features": 48,
  "total_active_features": 51,
  "word_coverage": 0.12,
  "char_coverage": 0.08,
  "total_feature_coverage": 0.10,
  "review_thresholds": {
    "minimum_decision_margin": 0.31,
    "minimum_word_coverage": 0.16,
    "minimum_char_coverage": 0.04,
    "minimum_total_coverage": 0.06
  }
}
```

## Field Semantics

| Field | Meaning |
| --- | --- |
| `predicted_label` | The model's forced best class: `E0`, `E1`, or `E23`. |
| `review_level` | `low`, `medium`, or `high`. `high` is the action threshold for human review. |
| `review_score` | Operational review risk from 0 to 1, derived from out-of-fold rarity of margin and vocabulary-coverage signals. It is not a class probability. |
| `needs_review` | Convenience boolean: `true` only when `review_level` is `high`. |
| `review_reasons` | Machine-readable reasons. Empty for accepted predictions. |
| `decision_margin` | Difference between the top and second-highest raw `LinearSVC` decision scores. It is not a probability. |
| `class_decision_scores` | Raw, uncalibrated decision scores for observability and later calibration. |
| `*_active_features` | Number of nonzero TF-IDF features activated by the input. |
| `*_coverage` | Fraction of candidate input n-grams that exist in the fitted TF-IDF vocabulary. Low word and character coverage is an OOV/style-shift signal. |
| `review_thresholds` | Exact deployed thresholds, echoed per result for auditability. |

## Review Reasons

- `low_decision_margin`: competing classes are close.
- `low_word_coverage`: few learned word n-grams matched.
- `low_char_coverage`: few learned character n-grams matched; strongest single signal of unfamiliar writing or terminology.
- `low_total_feature_coverage`: little sparse evidence overall.

Thresholds and score reference distributions are fitted on out-of-fold predictions, then frozen for deployment. The default `high` threshold targets roughly the riskiest 10% of familiar tasks, plus unfamiliar-writing/OOV cases. They are not training-set thresholds.

## Minimal API Contract

The public API returns only the fields a downstream decision flow needs:

```json
{
  "predicted_label": "E23",
  "review_level": "high",
  "review_score": 0.91,
  "needs_review": true
}
```

Set `include_diagnostics=true` for margins, vocabulary coverage, and review reasons. These fields are useful for debugging and monitoring, but should not be mistaken for calibrated probabilities.
