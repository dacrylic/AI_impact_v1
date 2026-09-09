# Faithful Appendix A.1 Factorial Rerun

## Objective

Re-estimate the original factorial study using the published Appendix A.1
rubric from *GPTs are GPTs*, rather than the shortened prompt used in the
earlier run. This rerun is required before interpreting prompt, model, or task
representation effects.

## Prompt construction

The system message will be the published Appendix A.1 exposure taxonomy in
`results/factorial_restart_v1/prompts/paper_2023_system_reconstruction.txt`.
That file must remain a faithful transcription of the paper's
`exposure_taxonomy.tex`; it must not be shortened or supplemented with custom
decision rules.

For every row, the only dynamic user content will be:

```text
Occupation: {occupation}
Task: {task}
Label (E0/E1/E2/E3):
```

The occupation is the retained job-role title. The task is either the raw
parent task or the linked atomic task, depending on the task-grain cell. Old
labels, new labels, scores, IDs, decomposition metadata, and any evaluation
information are excluded from the outbound request.

The experiment will not add a JSON schema, custom output instruction, or
post-hoc rubric language that was not part of the published prompt. Responses
will be stored verbatim and parsed only to identify the returned E0/E1/E2/E3
label. The raw response remains available for audit.

## Factorial structure

This is a `2 x 2 x 2` factorial, not `2 x 2 x 2 x 2`:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Rubric/prompt | Faithful Appendix A.1 paper rubric | Updated production rubric |
| Model | GPT-4o snapshot | GPT-5.2 snapshot |
| Task grain | Raw parent task | Linked atomic task |

The paper-prompt arm is the corrected arm for this rerun. The production arm
uses the existing updated production prompt so the prompt factor remains
defined. E2/E3 merging is a reporting transformation, not a factorial factor.
Historical labels are an external repeatability reference, not a factor.

## Data and calls

- Use the existing simple-random matched panel of 978 raw parents.
- Retain all 1,696 linked atomic children for those parents.
- Score every item in all four prompt/model cells.
- Expected calls: `(978 + 1,696) x 4 = 10,696`.
- Keep raw-parent and atomic-child outputs in separate analysis tables.
- Record requested model, resolved model snapshot, prompt hash, rendered request
  hash, timestamp, and raw model response for every call.

## Leakage controls

The panel is fixed before scoring. Sampling uses neither old labels nor new
labels. Each outbound request is checked to contain only occupation and task
text plus the fixed prompt. The analysis will verify:

- no duplicate call IDs;
- complete coverage of the expected 978 parents and 1,696 children;
- no target or score columns in rendered requests;
- identical prompt hashes within each prompt condition;
- raw and atomic records retain the correct parent-child linkage.

## Analysis plan

1. Report E0, E1, E2, E3 and the governance merge E23 = E2/E3 separately.
2. Compare prompt and model effects within the same grain using paired rows.
3. Compare raw and atomic results both as separate estimands and through a
   parent-level sensitivity analysis. Do not treat 1,696 children as 1,696
   independent parents.
4. For parent-level sensitivity, aggregate child labels using a declared rule
   and report alternatives. The primary rule will be modal class with ties
   resolved toward the lower ordinal category `E0 < E1 < E2 < E3`.
5. Use parent-cluster bootstrap intervals for raw-versus-atomic comparisons.
6. Compare each corrected paper-prompt cell with the historical labels only as
   a repeatability diagnostic, never as ground truth.

## Interpretation boundary

The rerun can establish how the published rubric behaves on this SFW panel
under current GPT-4o and GPT-5.2 snapshots. It cannot establish that the old
dashboard used this exact prompt or model, because the paper reports that the
GPT-4 prompt was modified to improve agreement with author labels and does not
publish the full historical API configuration.

The earlier results generated with the shortened prompt are invalid for the
factorial prompt comparison and must be marked superseded rather than pooled
with this rerun.
