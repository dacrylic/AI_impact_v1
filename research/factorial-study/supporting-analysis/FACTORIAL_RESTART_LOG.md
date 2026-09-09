# Factorial Study Restart Log

## 2026-08-29 restart

The first full factorial attempt was run from a temporary workspace
`/private/tmp/AI_impact_v1`.
It produced 3,302 successful teacher responses before the API account ran out
of credits. The temporary workspace was subsequently cleaned, so those
responses cannot be resumed.

The restart and its prevalence-panel continuation store artifacts in the
repository under `results/factorial_restart_v1/` and
`results/factorial_representative_v1/`:

- frozen prompt files and hashes;
- matched parent and atomic-child panels;
- label-free request manifest;
- append-only response checkpoint;
- transient-error rows and retry metadata;
- final analysis tables and management readout.

The analysis entry point is
`src/ai_impact_classifier/experiments/analyze_factorial_representative.py`.
It can be run against a partial checkpoint for interim diagnostics and again
after collection completes for the reportable frozen results. Its bootstrap
intervals are computed separately by grain and cell.

The production prompt is regenerated from the user-provided scorer source.
The paper condition is explicitly a reconstruction from the published rubric,
not a claim to possess the original tuned historical prompt.

The four prompt/model raw-parent cells and four atomic-child cells will be
run through the ordinary Responses API because the Batch API previously failed
at file authorization before processing any request. The runner uses the
project `.venv`, exponential retry/backoff, and resumes only successful call
IDs. API keys are loaded from the process environment and are never written to
this repository.
