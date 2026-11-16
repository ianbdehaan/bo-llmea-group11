import os
# one thread per process, the 10 runs of a candidate are evaluated in parallel processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import re
import sys
import random
import json
import shutil
import pickle
import argparse
from pathlib import Path
import yaml
import logging
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger


def clean_unfinished_generation(population, pop_dir):
    # When a search is resumed from the checkpoint, the candidates of the generation that was
    # running when it stopped are not in the checkpoint. They get generated again, so we move
    # their files to a separate folder (kept for transparency) and drop them from history.jsonl.
    if not pop_dir.exists():
        return
    known_ids = set(population.individuals.keys())
    n_known = len(known_ids)
    discard_dir = pop_dir / 'discarded_unfinished_generation'

    for f in pop_dir.iterdir():
        match = re.match(r'^(\d+)-(\d+)_', f.name)
        if f.is_file() and match and int(match.group(2)) > n_known:
            discard_dir.mkdir(exist_ok=True)
            shutil.move(str(f), discard_dir / f.name)

    history_path = pop_dir / 'history.jsonl'
    if history_path.exists():
        kept, dropped = [], []
        for line in history_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            if json.loads(line)['id'] in known_ids:
                kept.append(line)
            else:
                dropped.append(line)
        history_path.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')
        if dropped:
            discard_dir.mkdir(exist_ok=True)
            with open(discard_dir / 'history.jsonl', 'a', encoding='utf-8') as f:
                f.write('\n'.join(dropped) + '\n')
        print(f'Resumed: kept {len(kept)} candidates, moved {len(dropped)} unfinished ones to {discard_dir}')


def run_stage1_for_target(target_name, problem_id, instance_id, problem_desc, cfg):
    exp_cfg = cfg['experiment']
    gen_cfg = cfg['stage1_generation']
    llm_cfg = gen_cfg['llm']
    pop_cfg = gen_cfg['population']
    output_dir = Path(cfg['logging']['save_dir']) / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'checkpoint_population.pkl'

    print(f'Stage 1: {target_name} | problem {problem_id} | instance {instance_id}', flush=True)

    # 1. Evaluator: every candidate is run once for each of the 10 group seeds
    evaluator = IOHEvaluator(budget=gen_cfg['inner_budget'], dim=exp_cfg['dimension'], problems=[problem_id],
                             instances=[[instance_id]], repeat=len(exp_cfg['seeds']), seeds=exp_cfg['seeds'])
    # processes and not threads, because every run sets the global np.random seed
    evaluator.max_eval_workers = gen_cfg['eval_workers']
    evaluator.use_multi_process = gen_cfg['eval_workers'] > 0
    evaluator.timeout = gen_cfg['eval_timeout_seconds']

    # 2. Prompt generator (LLaMEA-BO baseline prompts with our reproducibility rule)
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    prompt_generator.problem_desc = problem_desc
    # "baseline": LLaMEA-BO prompts, "diverse": design space + random mutation operators
    prompt_generator.prompt_variant = gen_cfg.get('prompt_variant', 'baseline')
    prompt_generator.rng = random.Random(exp_cfg['seeds'][0])
    # the evaluator stops candidates that are too slow, and stage 2 runs the best one with 600 evaluations
    prompt_generator.runtime_prompt = (
        f"Each candidate is run {len(exp_cfg['seeds'])} times in parallel, one CPU core per run, and is stopped and counted as failed "
        f"if the runs take longer than {gen_cfg['eval_timeout_seconds']} seconds. Keep the cost per iteration small "
        f"(e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). "
        f"The algorithm will later also be run with a budget of {cfg['stage2_generalization']['evaluation_budget']} evaluations, "
        f"so it must work for any budget and its cost should grow moderately with the number of evaluated points."
    )

    # 3. LLM
    llm_manager = LLMmanager(model_name=llm_cfg['model_name'], base_url=llm_cfg['ollama_url'], num_ctx=llm_cfg['num_ctx'])
    llm_manager.max_api_retries = llm_cfg['max_api_retries']
    llm_manager.max_wait_seconds = llm_cfg['max_wait_seconds']

    # 4. Population: resume from the checkpoint if a previous run was interrupted
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            population = pickle.load(f)
        print(f'Resuming from {checkpoint_path}: generation {population.get_current_generation()}, '
              f'{population.get_population_size()} candidates done', flush=True)
        clean_unfinished_generation(population, Path(population.save_dir) / population._save_dir_suffix)
    else:
        population = ESPopulation(n_parent=pop_cfg['n_parent'], n_offspring=pop_cfg['n_offspring'],
                                  use_elitism=pop_cfg['use_elitism'])
        population.save_dir = str(output_dir)
        population.name = f"evol_{target_name}_dim{exp_cfg['dimension']}"
        population.debug_save_on_the_fly = True

    # 5. Run the evolution until outer_budget candidates are evaluated
    llm_params = {
        'temperature': llm_cfg['temperature'],
        'max_tokens': llm_cfg['max_tokens'],
        'timeout': llm_cfg['timeout_seconds'],
    }

    LLaMEvol().run_evolutions(
        llm=llm_manager,
        evaluator=evaluator,
        prompt_generator=prompt_generator,
        population=population,
        n_population=gen_cfg['outer_budget'],
        n_retry=pop_cfg['n_retry'],
        options={'llm_params': llm_params, 'checkpoint_path': str(checkpoint_path)},
    )

    # 6. Save the final population and the best algorithm (only one final file per search,
    # the plot script treats every final file as a separate search)
    pop_dir = Path(population.save_dir) / population._save_dir_suffix
    for old_final in pop_dir.glob('*_final_*.pkl'):
        old_final.unlink()
    population.save(suffix='_final')
    population.save_checkpoint(str(checkpoint_path))

    best_individual = population.get_best_of_all(maximize=True)
    if best_individual is None or best_individual.error:
        print('No working algorithm was found.')
        return
    handler = population.get_handler_from_individual(best_individual)
    safe_name = re.sub(r'[^\w\-.]', '_', handler.code_name or 'unnamed')
    # remove old best files, so stage 2 always picks up the current one
    for old in output_dir.glob(f'best_{target_name}_*.py'):
        old.unlink()
    best_code_path = output_dir / f'best_{target_name}_{safe_name}.py'
    with open(best_code_path, 'w', encoding='utf-8') as f:
        f.write(handler.code)
    print(f'Best algorithm saved to: {best_code_path}')
    print(f'Best Algorithm: {handler.code_name} (AOCC: {best_individual.fitness:.4f})')
    print(f'Evolution finished. Results saved in: {population.save_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['bbob', 'real_world', 'both'], default='both',
                        help='which LLaMEA search to run')
    parser.add_argument('--config', default='configs.yaml')
    args = parser.parse_args()

    setup_logger(level=logging.INFO, color=False)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.target in ('bbob', 'both'):
        bbob_cfg = cfg['targets']['bbob']
        run_stage1_for_target(
            target_name=f"bbob_f{bbob_cfg['function_id']}",
            problem_id=bbob_cfg['function_id'],
            instance_id=bbob_cfg['instance'],
            problem_desc=bbob_cfg['description'],
            cfg=cfg,
        )

    if args.target in ('real_world', 'both'):
        rw_cfg = cfg['targets']['real_world']
        run_stage1_for_target(
            target_name=f"real_world_{rw_cfg['problem_id']}",
            problem_id=rw_cfg['problem_id'],
            instance_id=rw_cfg['instance'],
            problem_desc=rw_cfg['description'],
            cfg=cfg,
        )


if __name__ == '__main__':
    main()
