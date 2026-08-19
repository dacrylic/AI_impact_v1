# New Target Initial Results

## Scope

These are CPU-only sparse classification screens on `tasks_reasoning_scores.csv`.
The canonical target is `E0`, `E1`, and `E23`, where source labels `E2` and
`E3` are merged. All results use five-fold `StratifiedGroupKFold` with
`jobrole_id` groups, so each holdout fold contains unseen roles.

The input contract is defined in [NEW_TARGET_DATA_CONTRACT.md](NEW_TARGET_DATA_CONTRACT.md).
In particular, no experiment uses `sector_title`, identifiers, scores, labels,
bands, or downstream rationale columns.

## Leakage Protocols

Two deliberately different strict protocols were used:

| Protocol | Input being protected | Preparation | Guaranteed zero overlap |
| --- | --- | --- | --- |
| Strict task-only | Normalized `task` | Remove task texts with conflicting canonical targets; retain one deterministic representative of each same-label task text | `jobrole_id` and task text |
| Strict full input | The exact model representation | Remove conflicting full inputs and retain one deterministic representative of each same-label full input | `jobrole_id` and full model-input string |

The task-only preparation contains 47,670 rows. It excludes 1,885 normalized
task texts (9,710 rows) with conflicting canonical labels. Same-label repeats
retain the row with the lexicographically smallest `jobrole_id`.

A mode-label sensitivity run retained the 1,121 conflicting task texts with a
clear majority label and removed 764 tied texts. It did not improve results,
supporting the stricter all-conflicts-excluded headline protocol.

Contextual representations are evaluated under strict full-input rather than
strict task-text protection because task text combined with a different job
title is a different permitted production input. The title-plus-structured
representation has no conflicting full-input labels and is reduced to 76,309
unique full inputs.

## Sparse Screen

All variants use word 1-2 gram TF-IDF plus character 3-5 gram TF-IDF and a
linear SVM. Scores below are pooled out-of-fold metrics.

| Representation | Leakage protocol | Rows | Accuracy | Macro F1 | Weighted F1 | Quadratic weighted kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Task only, unweighted | Strict task-only | 47,670 | 0.7826 | 0.7752 | 0.7821 | 0.6874 |
| Task only, balanced | Strict task-only | 47,670 | 0.7816 | 0.7757 | 0.7817 | 0.6894 |
| Task only, modal-conflict sensitivity | Strict task-only | 48,791 | 0.7778 | 0.7719 | 0.7778 | 0.6842 |
| Structured task: task + action/object/purpose | Strict full input | 47,699 | 0.7852 | 0.7777 | 0.7845 | 0.6926 |
| Title + task | Strict full input | 76,303 | 0.8219 | 0.8149 | 0.8213 | 0.7545 |
| Title + structured task | Strict full input | 76,309 | 0.8243 | **0.8177** | 0.8237 | 0.7604 |

The title-plus-structured variant was slightly stronger unweighted than with
`class_weight="balanced"` (0.8177 vs 0.8174 macro F1). The class-balanced
variant may still be useful when E1 recall is prioritised, but it is not the
current macro-F1 leader.

## Champion Candidate: Title + Structured Task

The current candidate uses this permitted text representation:

```text
ROLE: {jobrole_title}
TASK: {task}
ACTION: {action}
OBJECT: {object}
PURPOSE: {purpose}
```

Its pooled out-of-fold class metrics are:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.8510 | 0.8537 | 0.8523 | 20,891 |
| E1 | 0.7767 | 0.7394 | 0.7576 | 19,506 |
| E23 | 0.8330 | 0.8533 | 0.8430 | 35,912 |

It is still an initial screen, not a final deployment estimate: model and
feature selection have been informed by these folds. The next phase should
lock a candidate and assess it on a separate untouched role-held-out test set,
or use nested model selection.

## Interpretation

- The decomposition fields add a small but consistent improvement after role
  title is present (0.8149 to 0.8177 macro F1).
- The largest valid gain comes from permitted role-title context. This is
  consistent with identical task wording receiving different labels in
  different role contexts.
- Modal resolution of contradictory task-only labels is not an improvement;
  contradictions should remain a review/ambiguity signal rather than being
  silently forced into a class.
- The merged target retains the operational order `E0 < E23 < E1`. Quadratic
  weighted kappa is reported as an ordinal diagnostic only; the model is
  trained as a direct classifier in this phase.
