#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

uv run skydiscover-run \
  benchmarks/prompt_optimization/hotpot_qa/initial_prompt.txt \
  benchmarks/prompt_optimization/hotpot_qa/evaluator.py \
  -c benchmarks/prompt_optimization/hotpot_qa/config_adaevolve.yaml \
  -i 100
