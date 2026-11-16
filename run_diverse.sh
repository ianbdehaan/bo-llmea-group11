#!/bin/bash
# Alternative experiment with the "diverse" prompt variant (configs_diverse.yaml, results in outputs_diverse/).
# The baselines (BoTorch, CMA-ES) use exactly the same seeds and settings as in the main experiment, so their
# stage 2 runs are copied from outputs/ instead of being run again; only the two new algorithms are run.
cd "$(dirname "$0")"
mkdir -p logs_diverse outputs_diverse/stage2_benchmark/runs
PY=.venv/bin/python
CFG=configs_diverse.yaml

echo "$(date) stage 1 started"
$PY stage1_run.py --config $CFG --target bbob > logs_diverse/stage1_bbob.log 2>&1 &
PID_BBOB=$!
$PY stage1_run.py --config $CFG --target real_world > logs_diverse/stage1_real_world.log 2>&1 &
PID_RW=$!
wait $PID_BBOB; echo "$(date) stage 1 bbob finished (exit $?)"
wait $PID_RW; echo "$(date) stage 1 real world finished (exit $?)"

$PY check_reproducibility.py 60 $CFG > logs_diverse/reproducibility.log 2>&1
grep -v -i warn logs_diverse/reproducibility.log

cp outputs/stage2_benchmark/runs/*BoTorch_Vanilla* outputs/stage2_benchmark/runs/*CMA_ES* outputs_diverse/stage2_benchmark/runs/
echo "$(date) stage 2 started"
$PY stage2_benchmark.py --config $CFG > logs_diverse/stage2.log 2>&1
echo "$(date) stage 2 finished (exit $?)"

$PY run_analysis_plots.py $CFG > logs_diverse/plots.log 2>&1
$PY analyze_diversity.py outputs outputs_diverse > logs_diverse/diversity.log 2>&1
echo "$(date) plots and diversity analysis finished (exit $?)"
