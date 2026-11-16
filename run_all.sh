#!/bin/bash
# Runs the whole experiment: both LLaMEA searches in parallel, then the stage 2 benchmark and the plots.
# Usage: bash run_all.sh   (logs go to logs/). Finished parts are skipped/resumed when it is started again.
cd "$(dirname "$0")"
mkdir -p logs
PY=.venv/bin/python

echo "$(date) stage 1 started"
$PY stage1_run.py --target bbob > logs/stage1_bbob.log 2>&1 &
PID_BBOB=$!
$PY stage1_run.py --target real_world > logs/stage1_real_world.log 2>&1 &
PID_RW=$!
wait $PID_BBOB; echo "$(date) stage 1 bbob finished (exit $?)"
wait $PID_RW; echo "$(date) stage 1 real world finished (exit $?)"

echo "$(date) reproducibility check"
$PY check_reproducibility.py 60 > logs/reproducibility.log 2>&1
cat logs/reproducibility.log | grep -v -i warn

echo "$(date) stage 2 started"
$PY stage2_benchmark.py > logs/stage2.log 2>&1
echo "$(date) stage 2 finished (exit $?)"

$PY run_analysis_plots.py > logs/plots.log 2>&1
echo "$(date) plots finished (exit $?)"
