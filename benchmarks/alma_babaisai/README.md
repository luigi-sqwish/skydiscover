# ALMA BabaIsAI

SkyDiscover benchmark for optimizing a single ALMA `MemoStructure`
implementation on BabaIsAI.

## What It Does

- SkyDiscover mutates one Python memory file.
- The evaluator reads that file as text and sends it through ALMA's existing
  `Memo_Manager.execute_memo_structure(...)` path.
- ALMA remains unchanged and handles persistence, Docker execution, and score
  reporting.

This benchmark is intentionally scoped to **memory-only optimization**. It does
not co-optimize prompts or ALMA config files.

## Supported Split Modes

- `single_task`
- `multi_task`
- `generalization`

The populated ALMA split names for this benchmark are:

- `train`
- `eval_in_distribution`
- `eval_out_of_distribution` is currently empty upstream and should not be used
  for the default final score.

For every evaluation call, SkyDiscover preserves ALMA's task-specific rollout
and update protocol for the requested split.

## Prerequisites

1. Keep the repos side by side:
   - `.../skydiscover`
   - `.../alma`
2. Or set `ALMA_ROOT=/path/to/alma` if ALMA lives elsewhere.
3. Build ALMA's BALROG Docker image from the ALMA repo.
4. Export the model API key used by both SkyDiscover and ALMA.

## Run

From the `skydiscover` repo root:

```bash
uv run skydiscover-run benchmarks/alma_babaisai/initial_program.py \
  benchmarks/alma_babaisai/evaluator.py \
  -c benchmarks/alma_babaisai/config_adaevolve.yaml \
  --search adaevolve \
  --iterations 20
```
