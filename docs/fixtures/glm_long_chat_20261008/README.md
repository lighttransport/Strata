# Long document-chat evidence

See [the report and reproduction procedure](../../GLM_LONG_CHAT_QA.md).
Each context directory contains the frozen request, token-count/answer rubric,
actual final answer, full API response, runtime configuration, guard telemetry,
and manual assessment. `executed-runner.py` is an archived copy of the script
used for the ladder; use `tools/glm_long_chat_check.py` from the repository root
for reproduction. `run.json` records source/model/binary identity and limits.

- [16K answer](16384/answer.md) and [assessment](16384/assessment.json)
- [32K answer](32768/answer.md) and [assessment](32768/assessment.json)
- [64K answer](65536/answer.md) and [assessment](65536/assessment.json)
- [128K answer](131072/answer.md) and [assessment](131072/assessment.json)

These synthetic dossiers use the same eight factual questions at each length.
They are a retrieval/summarization smoke test, not broad chat qualification.

[Multi-turn follow-up answer](followup-32k/answer.md) and [assessment](followup-32k/assessment.json).
