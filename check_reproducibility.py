import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
import numpy as np
import yaml
from stage2_benchmark import load_generated_class, get_target_problem, run_generated_bo

# Runs every best algorithm from stage 1 twice with the same seed and checks that the
# two runs give exactly the same evaluations. Usage: python check_reproducibility.py [n_evals]

n_evals = int(sys.argv[1]) if len(sys.argv) > 1 else 60
cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else 'configs.yaml'))
dim = cfg['experiment']['dimension']
seed = cfg['experiment']['seeds'][0]
bbob = cfg['targets']['bbob']
rw = cfg['targets']['real_world']

all_ok = True
for algo_file in sorted(Path(cfg['logging']['save_dir']).glob('*/best_*.py')):
    algo_cls, name = load_generated_class(algo_file)
    for prob_id, inst in [(bbob['function_id'], bbob['instance']), (rw['problem_id'], rw['instance'])]:
        curve_1, _ = run_generated_bo(algo_cls, get_target_problem(prob_id, inst, dim), n_evals, dim, seed)
        curve_2, _ = run_generated_bo(algo_cls, get_target_problem(prob_id, inst, dim), n_evals, dim, seed)
        same = np.array_equal(curve_1, curve_2)
        all_ok = all_ok and same
        print(f'{name} on problem {prob_id}, seed {seed}: {"reproducible" if same else "NOT reproducible"} '
              f'(final best {curve_1[-1]:.6g} vs {curve_2[-1]:.6g})')

print('All algorithms reproducible.' if all_ok else 'Some algorithm is not reproducible, check its random generators.')
