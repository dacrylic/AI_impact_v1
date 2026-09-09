# Colleague Script Reproduction

## Purpose

Test whether the historical labels can be reproduced from the actual script
used by the original data scientist.

## Reproduced call

The call was reconstructed from the pasted script:

- System message: the script's full custom SFW rubric and five annotation
  examples.
- User message:

```text
Label the given task performed for the corresponding occupation with E0/E1/E2/E3.
Occupation: {jobrole}; Task: {task_statement}
Give your response in JSON format like this: {'label': 'final_lable'}
```

- Requested model: `gpt-4o`
- Temperature: `0.1`
- Seed: `6800`
- Response format: `{"type": "json_object"}`
- Inputs: occupation plus original raw task text
- Rows: the same 978 raw parents used in the factorial base panel

The internal gateway URL in the pasted script was not DNS-resolvable from this
environment. The reproduction therefore used the public OpenAI endpoint. The
public call resolved to `gpt-4o-2024-08-06`; the internal gateway's resolved
model remains unverified.

## Results

| | E0 | Impacted (E1/E2/E3) |
|---|---:|---:|
| Historical labels | 76.38% | 23.62% |
| Script reproduction | 76.99% | 23.01% |

Agreement against historical labels:

- Exact four-class agreement: **95.30%**
- E0-versus-impacted agreement: **96.11%**
- Macro F1: **0.9343**

Confusion matrix, historical labels as rows and reproduction labels as
columns:

| | Replay E0 | Replay E1 | Replay E2 | Replay E3 |
|---|---:|---:|---:|---:|
| Historical E0 | 731 | 11 | 5 | 0 |
| Historical E1 | 15 | 105 | 1 | 0 |
| Historical E2 | 7 | 7 | 92 | 0 |
| Historical E3 | 0 | 0 | 0 | 4 |

## Interpretation

This is strong evidence that the historical dataset was generated using a
prompt very close to the recovered script, at least for this base panel. It
also explains why the earlier Appendix A.1-only replays failed: the colleague's
five custom few-shot examples, exact user-message framing, JSON mode, seed, and
temperature materially constrain the model's behavior.

This is not absolute proof of the original run because the internal gateway,
exact server-side model snapshot, and any undocumented filtering or reruns are
not available. However, the 95.3% row-level agreement makes a hidden manual
massaging explanation unnecessary for the tested panel.

## Artifact

`results/colleague_reproduction_v1/responses_public.csv`
