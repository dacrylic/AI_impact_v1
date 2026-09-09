# Production Prompt With Legacy Anchors

This experiment adds a third prompt condition to the existing comparison:
the updated production rubric plus the five recovered SFW few-shot examples
from the legacy colleague script. The production user template and JSON output
contract remain unchanged.

The same 978 raw parents and 1,696 linked atomic children were used for both
models. All 5,348 calls completed successfully with no duplicate call IDs.

## Results versus production baseline

| Model / grain | Production E0 | Anchored E0 | Change | Production E23 | Anchored E23 |
|---|---:|---:|---:|---:|---:|
| GPT-4o / raw parent | 41.1% | 54.2% | +13.1 pp | 45.1% | 29.0% |
| GPT-4o / atomic child | 41.8% | 53.1% | +11.4 pp | 42.9% | 28.2% |
| GPT-5.2 / raw parent | 31.9% | 35.8% | +3.9 pp | 38.9% | 33.1% |
| GPT-5.2 / atomic child | 32.4% | 37.4% | +5.0 pp | 35.3% | 30.5% |

The anchored and production labels agree on 80.3% of GPT-4o raw parents and
85.2% of GPT-5.2 raw parents. The anchors make GPT-4o materially more E0-like,
while the effect on GPT-5.2 is smaller. This is evidence that the few-shot
examples are an additional substantive prompt factor, not a neutral formatting
change.

## Artifacts

- Prompt: `results/hybrid_anchor_prompt_v1/prompts/production_with_legacy_anchors_system.txt`
- Manifest: `results/hybrid_anchor_prompt_v1/manifest.csv`
- Responses: `results/hybrid_anchor_prompt_v1/responses.csv`
