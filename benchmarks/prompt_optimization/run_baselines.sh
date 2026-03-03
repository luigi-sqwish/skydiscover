#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

HOTPOT=benchmarks/prompt_optimization/hotpot_qa
PROGRAM="$HOTPOT/initial_prompt.txt"
EVALUATOR="$HOTPOT/evaluator.py"
ITERS=100

echo "Launching GEPA..."
nohup uv run skydiscover-run \
  "$PROGRAM" "$EVALUATOR" \
  -c "$HOTPOT/config_evox.yaml" \
  --search gepa --model gpt-5 -i "$ITERS" \
  > "$HOTPOT/gepa.log" 2>&1 &
echo "  GEPA PID: $!"

echo "Launching OpenEvolve..."
nohup uv run skydiscover-run \
  "$PROGRAM" "$EVALUATOR" \
  -c "$HOTPOT/config_evox.yaml" \
  --search openevolve --model gpt-5 -i "$ITERS" \
  > "$HOTPOT/openevolve.log" 2>&1 &
echo "  OpenEvolve PID: $!"

echo "Both running in background. Tail logs with:"
echo "  tail -f $HOTPOT/gepa.log"
echo "  tail -f $HOTPOT/openevolve.log"
