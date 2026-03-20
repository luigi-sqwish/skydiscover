# ALMA TextWorld Prototype

Prototype benchmark that lets SkyDiscover optimize a single ALMA `MemoStructure`
implementation for TextWorld.

## What It Does

- SkyDiscover mutates one Python memory file.
- The evaluator reads that file as text and sends it to ALMA's existing
  `Memo_Manager.execute_memo_structure(...)` path.
- ALMA remains responsible for saving the memory candidate, running the Docker
  benchmark, and returning the benchmark score.

This prototype is intentionally scoped to **memory-only optimization**. It does
not co-optimize separate prompt files or configs.

## Prerequisites

1. Keep the repos side by side:
   - `.../skydiscover`
   - `.../alma`
2. Or set `ALMA_ROOT=/path/to/alma` if ALMA lives elsewhere.
3. Build ALMA's TextWorld/BALROG Docker image from the ALMA repo.
4. Export the model API key used by both SkyDiscover and ALMA.

## Run

From the `skydiscover` repo root:

```bash
uv run skydiscover-run benchmarks/alma_textworld/initial_program.py \
  benchmarks/alma_textworld/evaluator.py \
  -c benchmarks/alma_textworld/config_adaevolve.yaml \
  --search adaevolve \
  --iterations 20
```

## Notes

- `single_task` and `multi_task` are supported.
- `generalization` is intentionally unsupported in this prototype.
- Agentic generation resolves its codebase from `ALMA_ROOT` when set, and
  otherwise falls back to the default sibling `../alma` layout used by the
  evaluator.
