# Action/Object/Purpose Stack Experiment

## Question

Can separate CPU-only specialists for `action`, `object`, and `purpose` add
useful signal beyond treating the decomposed activity as one task sentence?

## Target And Inputs

The target is `E0`, `E1`, and `E23`, with source E2 and E3 merged into E23.
The experiment uses only `action`, `object`, and `purpose`; it does not use
task text, job title, sector, IDs, scores, labels, bands, or rationale fields.

## Leakage Contract

The exact model input for this experiment is the normalized A/O/P tuple. Before
splitting, rows with label-conflicting tuples are removed and same-label repeats
are reduced to one deterministic representative. The outer five-fold split is
grouped by `jobrole_id`, and each fold must have both zero shared roles and
zero shared normalized A/O/P tuples.

Individual action, object, and purpose values are expected to repeat across
folds. That is not an exact-input leak: each specialist intentionally learns
generalizable associations from these component vocabularies. The full tuple is
the protected production input.

## Method

For each outer role-held-out fold:

1. Split the outer development roles once more into base-fit and meta-fit
   groups using `StratifiedGroupKFold`.
2. Fit independent word/character TF-IDF `LinearSVC` specialists on action,
   object, and purpose text using only base-fit rows.
3. Score the group-disjoint meta-fit rows and append a `purpose_present` flag.
4. Fit a regularized ridge-classifier meta-model on those compound scores.
5. Refit each specialist on the complete outer development partition, score
   the untouched outer test partition, and apply the meta-model.

This is holdout stacking, not nested hyperparameter search: all feature and
regularization choices are fixed before the outer test fold is scored. It is a
clean CPU method screen.

## Result

The strict A/O/P tuple preparation retained 47,699 unique, label-consistent
tuples. All five outer folds had zero shared job roles and zero shared A/O/P
tuples; the inner base-fit and meta-fit partitions also had zero shared roles.

The pooled out-of-fold result was 0.7574 accuracy, **0.7552 macro F1**, 0.7581
weighted F1, and 0.6517 quadratic weighted kappa. This is below the single
structured-task linear SVM on the same strict-input scale (0.7777 macro F1).

Conclusion: separate A/O/P specialists discard useful interactions that the
whole task sentence retains. This standalone stack is not a champion candidate.
