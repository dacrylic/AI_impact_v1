# Historical Old-Target Experiments

## Scope And Status

This document records experiments run against
`task_ai_impact_details (for sharing).xlsx`, the original task-level dataset.
It is a frozen historical record. The canonical deployed target in this phase
was `E0`, `E3`, and `E12`, with raw teacher labels `E1` and `E2` merged into
`E12`.

The original labels and scores were:

| Raw label | Source score | Meaning in source category |
| --- | ---: | --- |
| E0 | 0.1 | Enduring task |
| E3 | 0.3 | Automatable only by multimodal AI |
| E2 | 0.5 | Automatable by AI-powered software |
| E1 | 0.7 | AI automatable task |

The numerical scores were treated as teacher outputs, not inference features.
The source prompt and rubric were unavailable, so this work is knowledge
distillation rather than an independently validated AI-impact measure.

## Data And Evaluation Decisions

- The source workbook contained 36,905 rows (36,904 labeled rows), 1,858 job
  roles, and the text fields `keytask_content` and `jobrole_title`.
- Production input was restricted to `keytask_content` (required) and
  `jobrole_title` (optional). IDs, sector and occupation fields, source
  labels, scores, and categories were prohibited as features.
- Role context was investigated because identical task wording can receive
  different labels under different roles. It was permitted only when supplied
  as a title string, never as `jobrole_id`.
- The final valid benchmark is nested five-fold `StratifiedGroupKFold`, grouped
  by `jobrole_id`. All base-model and meta-model selection is performed within
  the outer development fold. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- Exact duplicate target text across distinct roles is reported as a warning,
  not a hard split failure: it can be a valid role-conditioned example. Shared
  job-role IDs are hard failures.

## Leakage Findings

Early exploratory results included nearly perfect accuracy for score-aided
models and a probability blend. These were invalid because the pseudo-target
score, or a label-derived equivalent, reached the feature path. They were
retained only as a diagnostic signal and excluded from every valid comparison.

The historical leaderboard also contains ordinary random train/validation/test
experiments. Those are useful for rapid method screening but are not comparable
with the group-held-out champion result and must not be used as a deployment
claim.

## Methods Explored

| Family | Representative implementation | Outcome |
| --- | --- | --- |
| Sparse text baselines | Word and character TF-IDF with LinearSVC / logistic regression | Strong CPU baseline; character-plus-word SVM was the most useful simple family. |
| Multi-view sparse stacks | Task-only and title-plus-task TF-IDF classifiers with a linear meta-model | Improved minority-class handling under role-grouped evaluation. |
| E3 specialists | E3-vs-rest sparse SVMs, retrieval similarity features, class offsets | Helpful in the final stack, though E3 remained the limiting class. |
| Rules and association mining | Keyword gates and association-rule overrides for E3 | Did not improve the selected out-of-fold stack; overrides often reduced macro F1. |
| Trees and classical alternatives | Decision tree, random forest, extra trees, histogram gradient boosting, naive Bayes, kNN retrieval | Below sparse linear SVM performance. |
| Dense embeddings | Frozen all-mpnet-base-v2 linear probe; dense+sparse stack | Frozen MPNet reached 0.6238 macro F1 on its benchmark; dense features did not justify CPU and latency cost. |
| Fine-tuned/cross-encoder screens | MPNet fine-tuning, DistilRoBERTa and DeBERTa cross-encoders | Explored as screens, not adopted; no result displaced the CPU sparse champion under the valid protocol. |
| Ordinal and score regression | Ridge, Huber SGD, and LinearSVR with validation-fitted thresholds | Underperformed direct classification. The nested canonical regression experiment produced fold macro F1 values from 0.5950 to 0.7080. |

The corresponding experiment scripts are retained under
`src/ai_impact_classifier/experiments/`. Result directories are intentionally
Git-ignored; the published claims and reproducible champion command are in the
README and model card.

## Selected Valid Results

The selected CPU-only champion is documented in [MODEL_CARD.md](MODEL_CARD.md):

| Metric | Outer-fold OOF result |
| --- | ---: |
| Accuracy | 0.9085 |
| Macro F1 | 0.7803 |
| Weighted F1 | 0.9090 |

Class-level F1 was 0.9402 for E0, 0.5850 for E3, and 0.8158 for E12. E3 had
201 examples and is the principal source of uncertainty. The five outer-fold
macro F1 values (0.7532, 0.7907, 0.8125, 0.7840, 0.7266) show material
role-held-out variation.

A simpler constrained sparse stack achieved 0.7711 macro F1 under the same
outer-CV setup. It remains the preferred fallback when the small incremental
gain of the champion does not justify its extra retrieval and specialist
features.

## Decision Record

The old-target phase ended with a CPU-first sparse solution because it had the
best valid macro F1 / complexity trade-off. It requires no GPU, embedding
service, Torch, or transformer model at inference.

The next dataset must be treated as a new target version. Its label definition,
row grain, and source-task lineage should be documented independently rather
than mixed into this historical benchmark.
