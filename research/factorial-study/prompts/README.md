# Teacher Prompt Archive

These files preserve the prompt inputs used to generate the teacher labels in
the factorial study. They are research lineage artifacts, not runtime
dependencies of the distilled classifier.

| Files | Condition |
| --- | --- |
| `production_system.txt`, `production_user_template.txt` | Current production guidance |
| `legacy_system.txt`, `legacy_user_template.txt` | Recovered legacy/paper-derived guidance |
| `production_with_legacy_anchors_system.txt`, `production_with_legacy_anchors_user_template.txt` | Current guidance with legacy few-shot anchors |

The user templates contain `{occupation}` and `{task}` placeholders used by
the factorial runner. Prompt hashes should be recorded with any new teacher
labelling run. The production model artifact in `models/` does not call an
LLM at inference time and does not load these files.
