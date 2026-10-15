# Production Output Schema

This is the canonical public response for both the FastAPI `/v1/predict` and
SageMaker endpoints. The API intentionally returns only three fields.

```json
{
  "predictedLabel": "E23",
  "reviewScore": 0.91,
  "modelVersion": "gpt52-current-guidance-e0-e1-e23-basic-v2"
}
```

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `predictedLabel` | string | Forced best class: `E0`, `E1`, or `E23`. `E23` merges source `E2` and `E3`. |
| `reviewScore` | number | Operational review-priority signal from `0` to `1`. Higher means the input is less familiar or the classifier decision is less clear. It is not a probability and is not a model confidence score. |
| `modelVersion` | string | Release identifier for the loaded artifact. |

## Review score semantics

The active predictor computes three signals:

- **Decision ambiguity:** a small gap between the top two `LinearSVC` decision
  scores increases review risk.
- **Lexical coverage:** a low fraction of task tokens found in the fitted task
  vocabulary increases review risk and acts as an OOV/style-shift signal.
- **Short input:** unusually short task text adds a small fixed risk.

The deployed score is the maximum of these risk components, clipped to `[0, 1]`
and rounded to four decimal places. It is a routing aid for downstream review,
not a calibrated estimate of label correctness. The current public API does
not return `needsReview`, `reviewLevel`, review reasons, decision margins, or
raw class scores.

## Input alternatives

Each request must use exactly one of these forms:

### AOP input

```json
{
  "action": "prepare",
  "object": "monthly operational performance report",
  "purpose": "support management review",
  "jobroleTitle": "Operations Analyst"
}
```

`action` and `object` are required for this form. `purpose` and `jobroleTitle`
are optional.

### Pre-composed task input

```json
{
  "keytaskContent": "prepare monthly operational performance report",
  "jobroleTitle": "Operations Analyst"
}
```

`keytaskContent` is required for this form and `jobroleTitle` is optional. Do
not send `keytaskContent` together with the AOP fields. The API does not
decompose free prose or infer missing AOP components.

## Batch output

`POST /v1/predict:batch` returns the same response object for each item:

```json
{
  "predictions": [
    {
      "predictedLabel": "E1",
      "reviewScore": 0.31,
      "modelVersion": "gpt52-current-guidance-e0-e1-e23-basic-v2"
    }
  ]
}
```

## Error shape

Validation, authentication, and model-readiness failures use this shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid request."
  }
}
```

The exact error code and message depend on the failure. Health endpoints are
public; scoring and model-information routes require `X-API-Key` on FastAPI.
SageMaker authentication and authorisation are supplied by the organisation's
gateway/IAM layer.
