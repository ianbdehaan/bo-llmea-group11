import os
# one thread per run, many runs in parallel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Stage 2 runs on the CPU only: the provided BoTorch baseline uses "cuda:3" when a GPU is visible,
# which does not exist on single-GPU machines, and small GPs are not faster on a GPU anyway
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import time
import random
import inspect
import argparse
import importlib.util
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
import cma
import ioh
from Practical_Problems.Practical_Problems_Suite import get_practical_problem
from baseline.botorch_vanilla import Vanilla_BO

from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

torch.set_num_threads(1)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    group_id = cfg['experiment']['group_id']
    configured_seeds = cfg['experiment']['seeds']
    expected_seeds = [10 * group_id + r for r in range(10)]
    assert configured_seeds == expected_seeds, f'Seed mismatch: group {group_id} needs {expected_seeds}, configs.yaml has {configured_seeds}'
    return cfg


def load_generated_class(py_file_path):
    spec = importlib.util.spec_from_file_location(py_file_path.stem, py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # the same rule as the LLaMEA code extraction: the first class whose name contains "BO"
    # (e.g. AdaptiveTrustRegionBO or AEBO_SB)
    match = re.search(r'class\s+(\w+BO\w*)\s*[:(]', Path(py_file_path).read_text(encoding='utf-8'))
    if match is None or not hasattr(mod, match.group(1)):
        raise AttributeError(f'No BO class found in {py_file_path}')
    return getattr(mod, match.group(1)), match.group(1)


def get_target_problem(problem_id, instance, dim):
    # ids >= 1000 are the practical (real-world) problems, the rest are BBOB functions
    if problem_id >= 1000:
        prob = get_practical_problem(problem_id, dim)
    else:
        prob = ioh.get_problem(problem_id, instance, dim)
    prob.reset()
    return prob


def run_generated_bo(algo_cls, problem, budget, dim, seed, time_limit=None):
    # the same seeding as in the stage 1 evaluator: np.random and torch are seeded before the run
    set_all_seeds(seed)

    best_so_far = []
    current_best = float('inf')
    start = time.time()
    status = {'value': 'ok'}

    def tracked_func(x):
        nonlocal current_best
        # stop the algorithm when it wants more evaluations than the budget
        if len(best_so_far) >= budget:
            raise RuntimeError('budget exhausted')
        # stop a run that takes longer than the time limit (keeps the benchmark within a few hours)
        if time_limit is not None and time.time() - start > time_limit:
            status['value'] = f'time limit at eval {len(best_so_far)}'
            raise RuntimeError(f'time limit of {time_limit} s reached')
        val = float(problem(np.asarray(x, dtype=np.float64).ravel()))
        current_best = min(current_best, val)
        best_so_far.append(current_best)
        return val

    optimizer = algo_cls(budget=budget, dim=dim)
    try:
        optimizer(tracked_func)
    except Exception as e:
        if status['value'] == 'ok' and len(best_so_far) < budget:
            status['value'] = f'error at eval {len(best_so_far)}: {type(e).__name__}'
        print(f'  run stopped at eval {len(best_so_far)} ({e})', flush=True)
    if status['value'] == 'ok' and len(best_so_far) < budget:
        status['value'] = f'stopped itself at eval {len(best_so_far)}'

    # if the algorithm stops early, its best value stays the same for the remaining budget
    while len(best_so_far) < budget:
        best_so_far.append(current_best)
    return np.array(best_so_far[:budget], dtype=float), status['value']


def run_botorch_vanilla(problem, budget, dim, seed, bo_cfg):
    set_all_seeds(seed)
    bounds = np.column_stack([problem.bounds.lb, problem.bounds.ub])
    opt = Vanilla_BO(budget=budget, n_DoE=bo_cfg['n_doe'], acquisition_function=bo_cfg['acquisition_function'], random_seed=seed)
    opt.verbose = False
    opt(problem=problem, dim=dim, bounds=bounds)

    best_history = np.minimum.accumulate(np.array(opt.f_evals, dtype=float))[:budget]
    if len(best_history) < budget:
        best_history = np.pad(best_history, (0, budget - len(best_history)), constant_values=best_history[-1])
    return best_history


def run_cma_es(problem, budget, dim, seed, cma_cfg):
    set_all_seeds(seed)
    lb = np.array(problem.bounds.lb)
    ub = np.array(problem.bounds.ub)

    # random start point in the box, sigma0 is a fraction of the box width
    x0 = np.random.uniform(lb, ub)
    sigma0 = float(np.mean(cma_cfg['sigma0_fraction'] * (ub - lb)))
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': [lb.tolist(), ub.tolist()],
        'seed': seed,
        'verbose': -9,
        'maxfevals': budget,
    })

    history = []
    best_y = float('inf')
    # restart with a new random point when CMA-ES stops before the budget is used
    while len(history) < budget:
        if es.stop():
            x0 = np.random.uniform(lb, ub)
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': [lb.tolist(), ub.tolist()], 'seed': seed + len(history),
                'verbose': -9, 'maxfevals': budget - len(history)})
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            if len(history) >= budget:
                break
            y = float(problem(x))
            fitnesses.append(y)
            best_y = min(best_y, y)
            history.append(best_y)
        if len(fitnesses) == len(solutions):
            es.tell(solutions, fitnesses)
    return np.array(history[:budget], dtype=float)


def run_one_task(task):
    # one run = one algorithm on one problem with one seed, executed in its own process
    torch.set_num_threads(1)
    problem = get_target_problem(task['problem_id'], task['instance'], task['dim'])

    start = time.time()
    if task['algo_type'] == 'generated':
        algo_cls, _ = load_generated_class(Path(task['algo_path']))
        curve, status = run_generated_bo(algo_cls, problem, task['budget'], task['dim'], task['seed'], task['time_limit'])
    elif task['algo_type'] == 'botorch':
        curve, status = run_botorch_vanilla(problem, task['budget'], task['dim'], task['seed'], task['bo_cfg']), 'ok'
    else:
        curve, status = run_cma_es(problem, task['budget'], task['dim'], task['seed'], task['cma_cfg']), 'ok'
    return task, curve, status, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg['experiment']['seeds']
    dim = cfg['experiment']['dimension']
    s2_cfg = cfg['stage2_generalization']
    budget = s2_cfg['evaluation_budget']
    assert s2_cfg['runs_per_algorithm'] == len(seeds)

    save_dir = Path(cfg['logging']['save_dir'])
    out_dir = save_dir / 'stage2_benchmark'
    run_dir = out_dir / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)

    # best algorithms from stage 1
    bbob_cfg = cfg['targets']['bbob']
    rw_cfg = cfg['targets']['real_world']
    bbob_name = f"bbob_f{bbob_cfg['function_id']}"
    rw_name = f"real_world_{rw_cfg['problem_id']}"
    bbob_scripts = sorted((save_dir / bbob_name).glob(f'best_{bbob_name}_*.py'))
    rw_scripts = sorted((save_dir / rw_name).glob(f'best_{rw_name}_*.py'))
    assert len(bbob_scripts) == 1, f'Expected one best_{bbob_name}_*.py in {save_dir / bbob_name}, found {len(bbob_scripts)}'
    assert len(rw_scripts) == 1, f'Expected one best_{rw_name}_*.py in {save_dir / rw_name}, found {len(rw_scripts)}'
    _, cls_name_bbob = load_generated_class(bbob_scripts[0])
    _, cls_name_rw = load_generated_class(rw_scripts[0])

    algorithms = {
        f'Generated_BBOB ({cls_name_bbob})': ('generated', str(bbob_scripts[0])),
        f'Generated_RealWorld ({cls_name_rw})': ('generated', str(rw_scripts[0])),
        'BoTorch_Vanilla': ('botorch', None),
        'CMA_ES': ('cmaes', None),
    }
    problems = {
        f"BBOB_F{bbob_cfg['function_id']}": (bbob_cfg['function_id'], bbob_cfg['instance']),
        f"RealWorld_{rw_cfg['problem_id']}": (rw_cfg['problem_id'], rw_cfg['instance']),
    }

    # every run is stored in its own csv, finished runs are skipped when the script is restarted
    tasks = []
    for prob_label, (prob_id, inst) in problems.items():
        for algo_label, (algo_type, algo_path) in algorithms.items():
            for seed in seeds:
                run_file = run_dir / f"{prob_label}__{algo_label.replace(' ', '_').replace('(', '').replace(')', '')}__{seed}.csv"
                if run_file.exists():
                    continue
                tasks.append({
                    'problem': prob_label, 'algorithm': algo_label, 'algo_type': algo_type, 'algo_path': algo_path,
                    'problem_id': prob_id, 'instance': inst, 'seed': seed, 'budget': budget, 'dim': dim,
                    'bo_cfg': s2_cfg['botorch'], 'cma_cfg': s2_cfg['cma_es'], 'run_file': str(run_file),
                    'time_limit': s2_cfg['run_time_limit_seconds'],
                })

    n_workers = max(1, min(s2_cfg['workers'], (os.cpu_count() or 2) - 1))
    print(f"Stage 2 | group {cfg['experiment']['group_id']} | dim={dim} | budget={budget} | seeds {seeds[0]}-{seeds[-1]}")
    print(f'{len(tasks)} runs left, {n_workers} parallel workers', flush=True)

    start_all = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_one_task, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                task, curve, status, run_time = fut.result()
            except Exception as e:
                print(f'  a run FAILED: {type(e).__name__}: {e}', flush=True)
                continue
            pd.DataFrame({
                'Problem': task['problem'], 'Algorithm': task['algorithm'], 'Seed': task['seed'],
                'Evaluation': np.arange(1, budget + 1), 'Best_So_Far': curve, 'Run_Time_s': run_time, 'Run_Status': status,
            }).to_csv(task['run_file'], index=False)
            print(f"  {task['problem']} / {task['algorithm']} / seed {task['seed']}: best {curve[-1]:.4e} [{status}] "
                  f"({run_time:.0f} s, total {(time.time() - start_all) / 60:.1f} min)", flush=True)

    # combine all runs of the current algorithms into one file (this is what the plot script reads)
    frames = []
    for prob_label in problems:
        for algo_label in algorithms:
            for seed in seeds:
                run_file = run_dir / f"{prob_label}__{algo_label.replace(' ', '_').replace('(', '').replace(')', '')}__{seed}.csv"
                if run_file.exists():
                    frames.append(pd.read_csv(run_file))
                else:
                    print(f'Missing run: {run_file.name}')
    results_df = pd.concat(frames, ignore_index=True)
    results_df.to_csv(out_dir / 'stage2_convergence_data.csv', index=False)

    # final performance: f(x) for the real-world problem, loss f(x) - f_opt for BBOB
    bbob_f_opt = float(ioh.get_problem(bbob_cfg['function_id'], bbob_cfg['instance'], dim).optimum.y)
    final = results_df[results_df['Evaluation'] == budget].copy()
    final['Final_Metric'] = np.where(final['Problem'].str.startswith('BBOB'), final['Best_So_Far'] - bbob_f_opt, final['Best_So_Far'])
    final['Metric_Type'] = np.where(final['Problem'].str.startswith('BBOB'), 'loss f - f_opt', 'objective f')
    summary_df = (final.groupby(['Problem', 'Metric_Type', 'Algorithm'])['Final_Metric']
                  .agg(mean='mean', std='std', se=lambda v: v.std(ddof=1) / np.sqrt(len(v)), median='median', min='min', max='max', runs='count')
                  .reset_index().sort_values(['Problem', 'mean']))
    summary_df.to_csv(out_dir / 'stage2_final_performance.csv', index=False)

    print(f'\nSaved: {out_dir}/stage2_convergence_data.csv')
    print(f'Saved: {out_dir}/stage2_final_performance.csv\n')
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
