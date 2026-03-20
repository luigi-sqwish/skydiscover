# Split-Aware Reference Benchmark

Minimal prompt-optimization example showing SkyDiscover's split-aware evaluator modes.

## Files

- `initial_prompt.txt`: seed prompt to evolve
- `evaluator.py`: split-aware Python evaluator using `split` and `phase`
- `config_multi_task.yaml`: evaluates and selects on `train`
- `config_generalization.yaml`: evaluates on `train` and `val`, but selects on `val`

## Run

```bash
uv run skydiscover-run \
  benchmarks/prompt_optimization/split_modes_reference/initial_prompt.txt \
  benchmarks/prompt_optimization/split_modes_reference/evaluator.py \
  -c benchmarks/prompt_optimization/split_modes_reference/config_multi_task.yaml \
  -s topk -i 5
```

```bash
uv run skydiscover-run \
  benchmarks/prompt_optimization/split_modes_reference/initial_prompt.txt \
  benchmarks/prompt_optimization/split_modes_reference/evaluator.py \
  -c benchmarks/prompt_optimization/split_modes_reference/config_generalization.yaml \
  -s topk -i 5
```

The evaluator intentionally rewards different behaviors on `train` and `val`, so `generalization` mode should preserve prompts that transfer better instead of prompts that only overfit the training split.
