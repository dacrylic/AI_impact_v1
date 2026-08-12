# CPU Classification Baseline Model Card

## Purpose

This CPU-first student approximates validated LLM labels. It predicts `E0`, `E1`, and `E23`, where `E23` combines the original teacher labels `E2` and `E3`.

It is a distillation model, not an independently validated measure of real-world AI impact. The originating LLM rubric and prompt are not available in this repository.

## Inputs And Exclusions

Required input: `keytask_content`.

Optional input used by the baseline: `jobrole_title`.

Excluded from all model features: IDs, occupation and sector titles/codes, `ai_impact_score`, categories, and source labels. `openai_label` is used only as the supervised training target.

## Method

[`run_constrained_oof_stack_cv.py`](../src/ai_impact_classifier/experiments/run_constrained_oof_stack_cv.py) cross-fits four word/character TF-IDF `LinearSVC` classifiers over task-only and title-plus-task views. A regularized multinomial logistic-regression meta-model combines their held-out scores.

## Evaluation

Nested five-fold `StratifiedGroupKFold` groups on `jobrole_id`. Each role is held out of its outer evaluation fold and inner model selection.

Aggregate out-of-fold results across 36,904 rows:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8938 |
| Macro F1 | 0.8056 |
| Weighted F1 | 0.8929 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.9363 | 0.9479 | 0.9421 | 28,009 |
| E1 | 0.7575 | 0.7100 | 0.7330 | 3,924 |
| E23 | 0.7496 | 0.7343 | 0.7419 | 4,971 |

Fold macro F1 values: 0.8057, 0.8012, 0.8022, 0.8155, and 0.8024.

## Limitations

- E1 and E23 are less frequent than E0 and are the limiting classes.
- Repeated task text can receive different labels across distinct job roles.
- This is the baseline for the corrected target, not a final selected champion.

## Intended Use

Use the model for batch triage or as a CPU-only proxy for the validated LLM labels. Do not use it as the sole basis for high-stakes employment or policy decisions without a separately documented rubric and human review process.
