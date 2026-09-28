#!/bin/bash
# Evaluate finished runs as they appear (skips runs already evaluated). Stops after the fine-tune queue reports FTDONE
# in runs/queue_ft.log; to evaluate a fixed set of runs once, call code/07_evaluate.py and code/07b_decompose.py directly.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
while true; do
  for m in runs/*/meta.json; do
    n=$(basename $(dirname $m)); case $n in _*|ft_*) continue;; esac
    if [ ! -f results/runs/$n/summary.json ]; then
      echo "=== eval $n $(date)"; python code/07_evaluate.py $n > results/runs_eval_$n.log 2>&1; python code/07b_decompose.py $n >> results/runs_eval_$n.log 2>&1; echo "=== done $n $(date)"
    fi
  done
  grep -q FTDONE runs/queue_ft.log 2>/dev/null && break
  sleep 300
done
echo EVALDONE
