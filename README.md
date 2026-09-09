# AI Impact Classifier

CPU-first ML distillation of the approved AI-impact scoring policy:

- **LLM policy:** GPT-5.2 with the approved current POC guidance.
- **Task input:** AOP is the primary API contract. Send `action`, `object`, and optional `purpose`; the API deterministically joins them into the same canonical task field used in training. It does not decompose prose.
- **Production target:** `E0`, `E1`, and `E23`, where source `E2` and `E3` are merged into `E23`.
- **Primary serving route:** FastAPI on Heroku.
- **Enterprise route:** a separate SageMaker endpoint deployment path using the same artifact.

This repository deliberately separates production code from data-science
evidence. Read [the experiments index](docs/EXPERIMENTS.md) for the evidence
behind the selected policy.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/ai_impact_classifier/production/` | Approved target contract, training and artifact-backed inference. |
| `src/ai_impact_classifier/api.py` | FastAPI application and camelCase public schema. |
| `deployment/heroku/` | Primary web API deployment instructions. |
| `deployment/sagemaker/` | Separate enterprise endpoint deployment path. |
| `research/factorial-study/` | Curated factorial-study documentation and approved presentation. |
| `src/ai_impact_classifier/experiments/` | Historical/reproducible experimentation scripts. Not serving code. |
| `artifacts/` | Local or release-managed model artifacts. Ignored from Git. |

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[api,dev]'
```

## Train the approved release candidate

The source file must contain the approved GPT-5.2/current-guidance labels and
the fields `jobrole_id`, `jobrole_title`, `task`, `action`, `object`,
`purpose`, and `label`.

```bash
ai-impact-train-production \
  --input "/path/to/tasks_reasoning_scores 2.csv" \
  --output-dir artifacts/releases/gpt52-current-field-aware-v1 \
  --refit-full
```

The command first creates a strict, role-grouped train/validation/sealed-test
split. Exact duplicate full model inputs are collapsed only when their target
is unanimous; contradictory duplicates are excluded rather than assigned an
arbitrary label. The selected model is evaluated on the sealed test, then
refit on all remaining approved rows only when `--refit-full` is supplied.

Review `training_report.json` before promoting `model.joblib` to
`artifacts/current/model.joblib` or publishing it to an approved artifact
store.

## Run the API

```bash
export AI_IMPACT_MODEL_PATH=artifacts/releases/gpt52-current-field-aware-v1/model.joblib
uvicorn ai_impact_classifier.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "prepare",
    "object": "monthly operational performance report",
    "purpose": "support management review",
    "jobroleTitle": "Operations Analyst",
  }'
```

The response is intentionally one stable schema:

```json
{
  "predictedLabel": "E1",
  "reviewScore": 0.31,
  "modelVersion": "gpt52-current-guidance-e0-e1-e23-v1"
}
```

`action` plus `object` are sufficient for the primary AOP route; `purpose` is
optional. `keytaskContent` is accepted as a compatibility alternative when
the upstream extractor already emits the composed task. The API never extracts
or infers A/O/P from prose.

`reviewScore` runs from `0` (little automatic review indicated) to `1`
(stronger review indicated). It combines classifier decision ambiguity,
lexical coverage and unusually short task text. It is an operational routing
signal, not a calibrated probability that the label is correct.

## Deployment

- [Heroku deployment](deployment/heroku/README.md)
- [SageMaker deployment](deployment/sagemaker/README.md)
- [Release and API guidance](docs/DEPLOYMENT.md)

## Guardrails

The model approximates an approved LLM scoring policy. It is not an
independently adjudicated measure of real-world AI impact. Do not send source
labels, scores, reasoning text, or identifiers to the prediction API. Monitor
novel language, review volume, class mix, and sampled disagreements after
release.
