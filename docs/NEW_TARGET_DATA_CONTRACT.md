# New Target Data Contract

## Scope

This document governs experiments on `tasks_reasoning_scores.csv`, the new
atomic-task dataset. It is separate from the historical workbook benchmark
documented in [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md).

Each row is a decomposed activity, represented by an `action`, `object`, an
optional `purpose`, and a composed `task`. It is not a one-row-per-original
SFw-key-task dataset.

## Canonical Target

The source label mapping for this phase is:

| Source label | Canonical label |
| --- | --- |
| E0 | E0 |
| E1 | E1 |
| E2 | E23 |
| E3 | E23 |

The resulting target distribution is 21,263 E0 rows (27.5%), 19,906 E1 rows
(25.7%), and 36,300 E23 rows (46.9%). This is a three-class classification
target. `score` is not used as a model input or regression target in the
initial benchmark.

The teacher labels retain a partial ordinal interpretation: `E0 < E3 < E2 <
E1`. After merging, `E23` is the middle band (`E3`/0.3 and `E2`/0.5), yielding
the operational order `E0 < E23 < E1`. The first benchmark remains direct
classification, with macro F1 as the primary selection metric, but also records
quadratic weighted kappa to audit the direction and severity of errors.

## Permitted Inputs

The initial CPU benchmark may use only:

- `task` (required)
- `action`
- `object`
- `purpose` when populated
- `jobrole_title` as an optional context string

`sector_title` is deliberately excluded because it may be unavailable at
production inference time. `jobrole_id` is a split key only, not a feature.

The following columns are forbidden as inference features:

- `label`, `band`, and `score`
- `reasoning_key` and `truncated_reason` (post-label rationales)
- `jobrole_id` and `task_no`
- `sector_title`

## Row Grain And Provenance

The new file does not retain the raw-task parent identifier (`cwfkt_id`) from
`latest_task_list.csv`. An original task can generate multiple atomic rows
with different labels. Direct exact text matching links only a subset of rows
to their source sentence; any later old-to-new reconciliation must either use
an exported parent ID or clearly label a reconstruction as inferred.

## Evaluation And Leakage Contract

The first benchmark uses five-fold `StratifiedGroupKFold` with `jobrole_id` as
the group. This tests generalisation to unseen job roles and makes a shared
role across development and evaluation rows a hard failure.

Exact task text can occur under multiple roles and can have different labels.
Those shared task strings are therefore reported as an audit warning, not
silently deduplicated or treated as a hard failure. A split that simultaneously
holds out every role and every repeated task is not viable here: the role-task
overlap graph has a giant component containing 82.9% of roles.

For the **task-only headline benchmark**, the runner uses a stricter policy:
it removes every normalized task string with more than one canonical target,
then retains one row for each remaining normalized task. Five-fold role-grouped
CV is run on that reduced set and any shared task text is a hard failure. This
prevents an exact task sentence from appearing in both training and evaluation
data. Same-label duplicate tasks retain the representative with the
lexicographically smallest `jobrole_id`, making that otherwise arbitrary
choice deterministic. Contextual models are evaluated separately because their input includes
role title, so identical task wording is not identical full model input. Their
strict benchmark instead deduplicates the complete normalized representation
and requires zero shared full model-input strings across folds.

The initial results are cross-validated method screens. A final selected model
must be locked before a separate untouched test set or a nested model-selection
protocol is used for a deployment claim.

## Initial Benchmark Grid

All methods are CPU-only sparse linear classifiers using word and character
TF-IDF with `LinearSVC`. They compare:

1. `task` only.
2. `task` plus labelled `action`, `object`, and `purpose` fields.
3. `jobrole_title` plus `task`.
4. The structured task representation plus `jobrole_title`.

Each representation is measured with ordinary and class-balanced training to
separate raw accuracy from minority-sensitive macro F1 behaviour. No sector,
score, label, rationale, identifier, embedding, or GPU feature is used.
