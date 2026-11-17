# Bayesian Optimization with LLaMEA-BO (Group 11)

This repository contains the implementation and evaluation for the **Bayesian Optimization 2026–2027 Practical Assignment** (Leiden University). We use LLaMEA (an LLM driven evolutionary algorithm) to generate Bayesian optimization algorithms for two 6-dimensional black-box problems, and then test how well each generated algorithm works on the other problem, compared with CMA-ES and a BoTorch BO baseline.

| | |
|:---|:---|
| Group | 11, seeds $s_{11,r} = 10 \cdot 11 + r = 110, \dots, 119$ |
| BBOB target | F8 (Rosenbrock, original), instance 1, $d = 6$ |
| Real-world target | Practical problem 1124 (parameter estimation of frequency modulated sound waves), $d = 6$ |
| Stage 1 | 100 candidate algorithms per search, 60 evaluations per run, 10 runs (seeds 110–119), ranked by AOCC |
| Stage 2 | 4 methods × 2 problems × 10 seeds × 600 evaluations, compared by final value and by AOCC |
| LLM | Qwen3-Coder-30B-A3B-Instruct (open weights, Q4_K_M) served locally with Ollama |
| Prompts | LLaMEA-BO prompts extended with a BO design space and random mutation operators (refine / replace a component / new idea) for more diverse algorithms (section 5.3) |

## 1. Repository structure

```text
├── configs.yaml               # all experiment settings (seeds, budgets, LLM, prompt, population, stage 2)
├── run_all.sh                 # runs everything: both stage 1 searches, reproducibility check, stage 2, plots
├── stage1_run.py              # stage 1: one LLaMEA search per target (resumable)
├── stage2_benchmark.py        # stage 2: 4 methods x 2 problems x 10 seeds (resumable)
├── check_reproducibility.py   # runs each best algorithm twice with the same seed and compares
├── run_analysis_plots.py      # stage 1 AOCC plots, stage 2 convergence plots and statistics
├── requirements.txt           # exact package versions
├── llamevol/                  # LLaMEA-BO framework (adapted, see section 6)
├── misc/                      # AOCC helpers from LLaMEA-BO
├── analysis/                  # plotting helpers from LLaMEA-BO
├── baseline/                  # BoTorch vanilla BO baseline (provided by the instructors)
├── Practical_Problems/        # real-world problem suite (provided by the instructors)
├── logs/                      # console logs of our runs (stage 1 searches, reproducibility check, stage 2, plots)
└── outputs/
    ├── bbob_f8/               # stage 1 search on F8: best algorithm, checkpoint, full search log
    ├── real_world_1124/       # stage 1 search on problem 1124: same structure
    ├── stage2_benchmark/      # stage 2: one csv per run (runs/), combined csv, final summary
    └── convergence_analysis/  # all figures and tables
```

Each stage 1 search folder `outputs/<target>/ESPopulation_<name>_<timestamp>/` contains for every candidate `<generation>-<index>_<name>_<AOCC>.py` (code), `_prompt.md` (system prompt + prompt), `_respond.md` (LLM answer + evaluation feedback or error) and `_handler.pkl`, plus `history.jsonl` with one line per candidate (generation, parents, prompt, response, code, error, AOCC of each of the 10 runs, run time, token counts) and the final population pickle.

The runs were made with the config file under the name `configs_diverse.yaml` and the output folder `outputs_diverse/`, which were renamed to `configs.yaml` and `outputs/` afterwards (only the `save_dir` line changed), so these names still appear in the logs.

## 2. Environment setup

Python 3.12 (3.11+ works). All optimization code runs on the CPU; a GPU is only used by the LLM server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU build of torch (the experiments do not use the GPU)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

With `uv` this is `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`.

`Practical_Problems/` is included in the repository (from https://github.com/BayesOptApp/Practical_Problems), so it is not installed separately.

numpy is pinned to 2.3.5: from numpy 2.4 on, `float()` of an array with shape `(1,)` raises a `TypeError`, and generated code does this very often (the code template stores `y` with shape `(n, 1)`, so `float(self.y[i])`); with 2.3 it is only a deprecation warning.

**CPU / GPU.** The implementation runs fully on CPUs. `stage2_benchmark.py` hides the GPU (`CUDA_VISIBLE_DEVICES=""`) because the provided BoTorch baseline selects the device `cuda:3` whenever a GPU is visible, which crashes on machines with fewer than four GPUs; for GPs with at most 600 points the CPU is fast enough. With the CPU build of torch, `torch.cuda.is_available()` is `False`, so the stage 1 prompt does not ask the LLM for GPU code either. The only hardware specific part is the LLM server (section 3).

## 3. Configure the LLM

We use **Qwen3-Coder-30B-A3B-Instruct** (open weights, free) served locally with Ollama: the Ollama model `qwen3-coder:latest`, a mixture of experts model with 30.5B parameters (3.3B active), quantized to Q4_K_M (≈ 19 GB; with a 32K context it needs ≈ 22 GB of GPU memory). No api key is needed and there are no rate limits. We chose it because it is free and open weight (as the assignment requires), specialised for code, answers without long hidden reasoning (so the answer is not cut off before the code), and fast enough on one RTX 3090 (≈ 85 tokens/s) to run both searches within a few hours; it was released in July 2025, so newer models exist, but the choice was limited to models that run locally on the available hardware.

```bash
# on the machine with the GPU (https://ollama.com/download)
ollama pull qwen3-coder
ollama serve            # usually already running as a service on localhost:11434
```

The model is configured in `configs.yaml` under `stage1_generation.llm`: `model_name` (Ollama model name), `ollama_url` (server address) and `num_ctx` (context length); for models with a thinking mode (e.g. `qwen3`) it can be switched on or off with `think: true/false`. `llamevol/llm.py` talks to the native Ollama api (`/api/chat`) and passes the context length explicitly, because the OpenAI compatible endpoint of Ollama silently uses a short default context and would cut long prompts. Any other model available in Ollama can be used by changing `model_name`.

## 4. Running the experiments

Everything (both searches in parallel, the reproducibility check, stage 2 and the plots) in one command, logs go to `logs/`:

```bash
bash run_all.sh
```

Or step by step:

```bash
python stage1_run.py --target bbob          # LLaMEA search on BBOB F8
python stage1_run.py --target real_world    # LLaMEA search on problem 1124
python check_reproducibility.py             # same seed twice -> same result for both best algorithms?
python stage2_benchmark.py                  # generalization benchmark
python run_analysis_plots.py                # figures and tables in outputs/convergence_analysis/
```

All scripts read `configs.yaml`; another config file can be given with `--config <file>` (stage 1 and 2) or as the last argument (`run_analysis_plots.py <file>`, `check_reproducibility.py <n_evals> <file>`).

`run_analysis_plots.py` writes to `outputs/convergence_analysis/`:

| File | Content |
|:---|:---|
| `stage1_aocc_history.png/pdf` | AOCC of every candidate in generation order (failed ones as AOCC 0) and the best AOCC so far, both searches |
| `stage1_summary.csv` | candidates, failures, best candidate and AOCC, token usage per search |
| `stage1_<target>/es_aoc_lines.png`, `search_aoc_voilin.png`, `overall_error_rate_voilin.png` | LLaMEA-BO plots: AOCC per generation, distribution, error rate |
| `stage2_bbob_loss_convergence.png/pdf` | BBOB: best-so-far loss $f - f_{opt}$ vs evaluations, mean ± standard error over the 10 runs, linear axes; left the whole run, right a zoom on evaluations 60–600 |
| `stage2_realworld_objective_convergence.png/pdf` | real-world problem: best-so-far objective vs evaluations, same layout |
| `stage2_statistics.csv` | per method and problem: the final value after 600 evaluations (mean, std, SE, median, min, max) and the AOCC of the runs (mean, std, median), each with a two-sided Wilcoxon rank-sum test (10 vs 10 runs) against the best method (best median), Holm corrected per problem |

`outputs/stage2_benchmark/stage2_final_performance.csv` (written by stage 2) has the final value summary without the tests.

Both stages can be interrupted and started again with the same command: stage 1 continues from `outputs/<target>/checkpoint_population.pkl` (saved after every generation), stage 2 skips runs whose csv already exists in `outputs/stage2_benchmark/runs/`. To start a search from scratch, delete its `outputs/<target>/` folder.

Stage 1 writes the best algorithm to `outputs/<target>/best_<target>_<ClassName>.py`; stage 2 picks these two files up automatically.

## 5. Configuring prompts and hyperparameters

### 5.1 Settings

All settings are in `configs.yaml`:

| Section | Parameter | Value | Meaning |
|:---|:---|:---|:---|
| experiment | `group_id`, `seeds` | 11, 110–119 | stage 2 checks that `seeds == [10*group_id + r]` |
| experiment | `dimension` | 6 | |
| targets.bbob | `function_id`, `instance` | 8, 1 | BBOB F8 Rosenbrock, instance 1 |
| targets.real_world | `problem_id` | 1124 | FM sound waves |
| targets.* | `description` | text | problem description inserted in the prompt |
| stage1_generation | `outer_budget` | 100 | candidate algorithms per search |
| stage1_generation | `inner_budget` | 60 | evaluations per run (10·d), including the initial design |
| stage1_generation | `prompt_variant` | `diverse` | `diverse`: design space + random mutation operators (section 5.3); `baseline`: the original LLaMEA-BO prompts |
| stage1_generation | `eval_workers` | 10 | the 10 runs of a candidate in parallel processes |
| stage1_generation | `eval_timeout_seconds` | 120 | a candidate whose 10 runs take longer fails |
| population | `n_parent`, `n_offspring`, `use_elitism` | 4, 12, true | (4 + 12) elitist ES of LLaMEA-BO |
| population | `n_retry` | 3 | re-prompts if the answer contains no code |
| llm | `model_name`, `ollama_url`, `num_ctx` | `qwen3-coder:latest`, `http://localhost:11434`, 32768 | see section 3 |
| llm | `temperature` | 0.8 | sampling temperature |
| llm | `max_tokens` | 16384 | max generated tokens per answer |
| llm | `timeout_seconds`, `max_api_retries`, `max_wait_seconds` | 900, 30, 3600 | request timeout and retry/backoff of failed requests |
| stage2_generalization | `evaluation_budget` | 600 | evaluations per run (100·d), including the initial design |
| stage2_generalization | `workers` | 40 | parallel runs |
| stage2_generalization | `run_time_limit_seconds` | 10800 | safety limit: a generated algorithm run is stopped after 3 h (status in the csv); no run reached it |
| stage2_generalization.botorch | `n_doe`, `acquisition_function` | 12, EI | initial design of 2·d points, expected improvement |
| stage2_generalization.cma_es | `sigma0_fraction` | 0.2 | σ0 = 0.2·(ub − lb) = 2, uniform random x0, restarts if CMA-ES stops early |

### 5.2 Prompts

The prompts are built in `llamevol/prompt_generators/bl_prompt_generator.py` (`BaselinePromptGenerator` with `is_bo = True`, the LLaMEA-BO prompt). The task description is `__bo_task_description`, the code template is `__bo_code_structure`, the answer format (`# Description / # Justification / # Code`) is `response_format`, and the feedback after an evaluation is `evaluation_feedback_prompt` (mean and std of the AOCC and the run time). Two parts are set from `stage1_run.py`: `reproducibility_prompt` and `runtime_prompt` (section 6). Mutation prompts include the parent code and a summary of the current population.

### 5.3 Prompts for more diverse algorithms

With the original LLaMEA-BO prompts the LLM tends to produce the same kind of algorithm again and again: every mutation asks *"Refine the strategy of the selected solution to improve it."*, so the offspring are small variations of their parent, and nothing in the prompt suggests alternatives to the default choices of the model (typically a scikit-learn GP with expected improvement). To make the search explore different designs we changed two things (`prompt_variant: "diverse"`, `DESIGN_SPACE`, `DIVERSE_MUTATIONS`, `__design_space_prompt` and `__diverse_mutation` in `llamevol/prompt_generators/bl_prompt_generator.py`). The temperature (0.8) and everything else is unchanged.

**1. Design space in the task description.** Every prompt (initial generation and mutations) contains this list after the problem description:

```text
A BO algorithm can be built from many different components, for example:
- initial design: Sobol, Latin hypercube, random, maximin designs, different sizes (e.g. d+1 up to 4d points)
- surrogate model: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y,
  random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted
  only near the best point
- acquisition function: expected improvement, log expected improvement, lower confidence bound with a schedule for
  the exploration weight, probability of improvement, Thompson sampling from the GP posterior, a portfolio that
  switches between several acquisition functions
- acquisition function optimizer: L-BFGS-B from several starting points, differential evolution, a small evolution
  strategy, random candidates plus local refinement, candidates sampled around the best points
- exploration strategy: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when
  progress stops, a switch from global to local search, batches of diverse points
Do not always use the same default choices (e.g. a scikit-learn GP with expected improvement on random candidates);
choose the components that suit the problem.
```

**2. Random mutation operators.** Instead of always "refine", every mutation prompt ends with one of three operators, chosen uniformly at random:

| Operator | Text in the prompt | Purpose |
|:---|:---|:---|
| refine | *"Refine the strategy of the selected solution to improve it."* (the original text) | local improvement of a good design |
| replace component | *"Replace the {component} of the selected solution by a substantially different alternative (options: {options}), and adapt the rest of the algorithm where needed. Explain in the justification why the new {component} should work better."* The component is chosen uniformly from the five components above, and its options from the list are inserted. | structural change of one part of the algorithm |
| new idea | *"Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification."* | jump to a different design (the prompt also contains a summary of the current population) |

The operators are drawn with Python's `random.Random(110)` (seeded with the first seed of the group), so the sequence of operators is reproducible and the same in both searches; the operator of every candidate can be read in its saved prompt (`_prompt.md`, `history.jsonl`). With `prompt_variant: "baseline"` the original LLaMEA-BO prompts are used.

**How often they were used and what they gave** (the 96 offspring after generation 0 of each search; the same operator sequence in both searches):

| Operator | Used | Working offspring (BBOB / real-world) | Better than the parent (BBOB / real-world) |
|:---|:---:|:---:|:---:|
| refine | 40 | 25 / 33 | 5 / 11 |
| replace component | 29 (exploration strategy 8, surrogate 7, acquisition function 5, acquisition optimizer 5, initial design 4) | 20 / 20 | 7 / 9 |
| new idea | 27 | 6 / 19 | 2 / 1 |

"Replace component" is the most successful operator relative to how often it was used, and the key steps towards both best algorithms were made by it: the BBOB lineage switched to a trust region ("replace the exploration strategy"), the real-world lineage to a random forest and then a new initial design (section 9.1). "New idea" mostly produced broken code on BBOB (21 of 27 failed).

### 5.4 Selection metric (AOCC)

Every candidate is run once per seed (110–119) with 60 evaluations. For each run the evaluator (`ConvergenceCurveAnalyzer` in `llamevol/evaluator/evaluator_result.py`, as in LLaMEA-BO) takes the best-so-far curve of the loss $f(x_t) - f_{opt}$, clips it to $[10^{-8}, 10^{9}]$ (LLaMEA-BO uses the upper bound $10^9$ for $d > 5$), scales it on a log10 scale to $[0, 1]$ and computes

$$\text{AOCC} = 1 - \int_0^1 \frac{\log_{10} \text{clip}(\min_{s \le t} f(x_s) - f_{opt}) - \log_{10} 10^{-8}}{\log_{10} 10^{9} - \log_{10} 10^{-8}} \, dt$$

with the trapezoidal rule over the evaluations (1 is best). The fitness of a candidate is the mean AOCC over the 10 runs; candidates that crash, exceed the budget or time out get AOCC 0. For BBOB F8, $f_{opt}$ is the known optimum of instance 1 ($f_{opt} = 149.15$, `ioh.get_problem(8, 1, 6).optimum.y`). For problem 1124 the optimum is unknown and the suite uses $f_{opt} = 0$ (the objective is a sum of squared errors between the generated and the target sound wave, so it is non-negative and 0 means a perfect fit), so its AOCC is computed on the raw objective. The best algorithm of each search is selected only by its AOCC on its own target. In stage 2 the same AOCC is computed for every run with 600 evaluations (`run_analysis_plots.py`), so the methods are also compared on their anytime performance.

### 5.5 Hyperparameters of the generated algorithms

Their hyperparameters (initial design size, kernel, acquisition settings, trust region constants) are constants in `__init__` of the generated files `outputs/<target>/best_*.py` and can be edited there.

## 6. Modifications to LLaMEA and why

The `llamevol/` package is LLaMEA-BO (see credits). All our changes are marked with `# Group 11 change` in the code.

1. **LLM client** (`llamevol/llm.py`). The original file has clients for many remote api providers and uses OpenAI compatible endpoints. We replaced them with a single client for the native Ollama api (`/api/chat`), because the OpenAI compatible endpoint of Ollama silently uses its short default context length (our prompts have up to ≈ 10K tokens); the native api lets us set it (`num_ctx`), and also the thinking mode of models that have one (`think`). The client also retries a failed request with exponential backoff instead of turning it into a failed candidate; this was meant for remote apis and was never needed with the local server (0 retries in both searches).
2. **Reason of a failed answer in the logs** (`llamevol/llamevol.py`). When an answer contains no code, the original code only logged that no code was found. Now the reason (request error, finish reason such as `length` when `max_tokens` is used up, empty answer, or no code block) is recorded in the error of the candidate, so it appears in `_respond.md` and `history.jsonl`. In our searches this happened once (BBOB), and the automatic re-prompt gave a valid answer. For models with a thinking mode the reasoning is saved before the answer.
3. **Complete search history** (`population.py`). Every candidate is appended to `history.jsonl`, and the population is checkpointed after every generation so a search can be resumed (`stage1_run.py` moves the files of an unfinished generation to `discarded_unfinished_generation/`).
4. **Budget check** (`ioh_evaluator.py`). The over-budget check used `> budget`, which allowed 61 evaluations in a 60 evaluation run; now `>= budget`.
5. **Timeout really stops a candidate** (`ioh_evaluator.py`). After a timeout the executor waited for the running processes, so a hanging candidate blocked the search. Now the worker processes are terminated. Together with `eval_timeout_seconds` this keeps one search at a few hours.
6. **Parallel evaluation in processes** (`stage1_run.py`). The LLaMEA-BO evaluator can run the 10 runs of a candidate one after another, in threads or in processes; we use processes (`use_multi_process`, 10 workers). The evaluator seeds the global `np.random` state with the seed of the run, and generated code draws from this state (directly or to create its own generators), so the runs cannot simply be run in threads of one process: the threads would share one global state, every `np.random.seed` call would overwrite the seed of the other runs, and which run gets which random numbers would depend on the timing, so the same seed would no longer give the same result. We run the 10 runs in 10 separate processes instead, each with its own `np.random` state: this is ≈ 10 times faster than running them one after another and just as reproducible.
7. **Reproducibility rule in the prompt** (`reproducibility_prompt`). The evaluator seeds `np.random` and `torch` before each run, but generated code often creates unseeded generators (e.g. `qmc.Sobol(scramble=True)`), so the same seed gave different results. The prompt now asks to derive every generator from `np.random`; `check_reproducibility.py` confirms that both best algorithms are reproducible.
8. **Run time rule in the prompt** (`runtime_prompt`). Tells the LLM about the timeout and that the algorithm is later run with 600 evaluations, so it should not become too expensive per iteration.
9. **Problem description.** The original prompt said "evaluated on the BBOB test suite of ..." also for the real-world problem; it now uses the description from `configs.yaml`.
10. **Prompts for more diverse algorithms** (`bl_prompt_generator.py`, section 5.3). Design space in the task description and random mutation operators.
11. **Real-world problems and seeds in the evaluator** (`ioh_evaluator.py`). LLaMEA-BO only supports BBOB functions; problem ids ≥ 1000 now load the practical problem from `Practical_Problems/`, dimension 6 is allowed, and every run seeds `np.random` and `torch` with its own seed (110–119) instead of using unseeded runs.

## 7. Randomness and reproducibility

* Every run (stage 1 and 2) seeds `random`, `np.random` and `torch` with its seed from 110–119 before the algorithm is created. BoTorch gets `random_seed=seed`, CMA-ES `seed=seed` (and `seed + evaluations used` on a restart).
* `check_reproducibility.py` runs both best algorithms twice per problem with seed 110 and checks that the results are identical; both are reproducible (`logs/reproducibility.log`).
* The choice of the mutation operator uses its own seeded generator (`random.Random(110)`).
* **Not controlled:** the LLM sampling (temperature 0.8, no seed), so a new search gives different algorithms. Floating point results of BLAS libraries can differ between machines/library versions.

## 8. Machines

| Machine | Used for | Hardware | Software |
|:---|:---|:---|:---|
| vibranium.liacs.nl (LIACS REL Compute) | stage 1 (LLM + evaluation), stage 2 | 2× Intel Xeon Silver 4214R @ 2.40 GHz (48 threads), 251 GB RAM, 2× NVIDIA GeForce RTX 3090 (24 GB), driver 615.71 | Rocky Linux 9.8, Python 3.12, torch 2.15.0+cpu, Ollama 0.32.5 |
| local PC | final plots and tables | AMD Custom APU 0405 (8 threads) | Arch Linux, the same Python packages (`requirements.txt`) |

Run times on vibranium: the two stage 1 searches ran in parallel (10 cores each, sharing the LLM) and took 1 h 25 min (BBOB) and 2 h 23 min (real-world). Stage 2 of the two generated algorithms took 34 min with 40 parallel runs. The BoTorch and CMA-ES runs do not depend on the LLM; they were computed once with the same `stage2_benchmark.py`, seeds and settings (≈ 1 h 15 min in total, a BoTorch run takes ≈ 21 min) and are used as they are.

The machine is shared with other users. The LLM was served by the machine's Ollama service on its GPUs; all optimization runs used one CPU core each (`OMP_NUM_THREADS=1`).

## 9. Generated algorithms

### 9.1 The two searches

| | BBOB F8 search | Real-world 1124 search |
|:---|:---|:---|
| candidates (generations) | 100 (4 + 8 × 12) | 100 (4 + 8 × 12) |
| failed candidates | 46 (22 syntax errors, 7 ValueError, 7 NameError, 4 over budget, 3 IndexError, 2 timeouts, 1 AttributeError) | 27 (11 timeouts, 8 syntax errors, 2 ValueError, 2 IndexError, 2 TypeError, 2 NameError) |
| best AOCC per generation 0, 1, …, 8 | 0.339, 0.345, **0.369**, 0.367, 0.369, 0.367, 0.368, 0.366, 0.368 | 0.427, 0.451, 0.462, 0.456, 0.462, **0.470**, 0.468, 0.470, 0.470 |
| best algorithm | `AdaptiveTrustRegionBO` (candidate 44, generation 4, AOCC 0.369) | `EMBO_SB` (candidate 57, generation 5, AOCC 0.470) |
| lineage of the best (operator) | #1 `EfficientRobustBO` (gen 0, 0.338) → #15 `AdaptiveMultiSurrogateBO` (refine, 0.342) → #21 `AdaptiveTrustRegionBO` (replace exploration strategy, 0.369) → #44 (refine, 0.369) | #2 `EMBO` (gen 0, 0.427) → #16 `EMBO` (new idea, 0.427) → #17 `EMBO_SB` (refine, 0.456) → #46 `EMBO_RF` (replace surrogate model, 0.459) → #57 `EMBO_SB` (replace initial design, 0.470) |
| LLM tokens (prompt / answer) | 363K / 261K | 416K / 293K |
| wall clock time | 1 h 25 min | 2 h 23 min |

On BBOB most failures are syntax errors, on the real-world problem most failures are timeouts (candidates whose 10 runs take longer than 120 s). Both searches found their best algorithm in the middle of the search (generations 4 and 5); the AOCC improved by +0.030 (BBOB) and +0.043 (real-world) over the best candidate of generation 0.

### 9.2 BBOB F8: `AdaptiveTrustRegionBO`

File: `outputs/bbob_f8/best_bbob_f8_AdaptiveTrustRegionBO.py`

```text
n_init = min(max(10, 2d+1), budget/4) = 13 LHS points, radius r = 1
loop until the budget is used:
    GP <- fit GP(Matern 5/2, one length scale, + white noise), normalize_y, 2 restarts
    if the best value improved by > 0.5 %: r <- min(1.2 r, 5)
    else after 3 iterations without improvement: r <- max(0.5 r, 0.01)
    if r reached 0.01 without improvement: r <- 1       # "restart", around the same best point
    C <- min(20d, 500) = 120 uniform points in the box [x_best - r, x_best + r] (clipped)
    evaluate the point of C with the largest expected improvement
return best (y, x)
```

| Component | Choice | Compared with a standard BO pipeline |
|:---|:---|:---|
| Initial design | 13 LHS points (22 % of the 60 evaluation budget) | standard |
| Surrogate | sklearn GP, Matern ν = 5/2 with a single length scale + white noise, `normalize_y`, 2 optimizer restarts | standard; the comment in the code calls the kernel "ARD", but it uses one length scale |
| Acquisition | expected improvement | standard |
| Acquisition optimizer | argmax over 120 random points in the trust region | no gradient optimizer; the trust region does the local refinement |
| Exploration strategy | trust region around the incumbent that grows after an improvement and shrinks after 3 iterations without one, reset to r = 1 at the minimum size | TuRBO-like local BO, but the "restart" stays around the same best point (TuRBO restarts from a new design) |

### 9.3 Real-world 1124: `EMBO_SB`

File: `outputs/real_world_1124/best_real_world_1124_EMBO_SB.py`

```text
n_init = max(2d+1, 10) = 13 Sobol points
loop until one evaluation is left:
    RF <- random forest (50 trees, depth 10) on all points
    batch <- min(5, remaining/4) (2-3 at the end)
    x1 <- among 10-20 Sobol points, the one with the lowest RF prediction     # see below
    x2..x_batch <- x_best + N(0, 0.2^2) (clipped)                            # local search
    evaluate the batch
return best (y, x)
```

| Component | Choice | Compared with a standard BO pipeline |
|:---|:---|:---|
| Initial design | 13 Sobol points | standard |
| Surrogate | random forest (50 trees, depth 10) | SMAC-like surrogate, cheap (≈ 4 min per 600 evaluation run) |
| Acquisition | described in the code as a lower confidence bound, but the "uncertainty" is the range of all observed values, the same for every point, so it only ranks by the RF mean | effectively the RF mean (pure exploitation of the model) |
| Acquisition optimizer | L-BFGS-B on the *negative* of this value (a sign error), but the RF prediction is piecewise constant, so L-BFGS-B stops at its starting point and the sign has no effect: in practice the best of 10–20 Sobol points by RF prediction | screening of random points |
| Exploration strategy | most of every batch are Gaussian perturbations (σ = 0.2) of the best point | local search around the incumbent |

It stops one evaluation before the budget (599 of 600).

### 9.4 Differences between the two

Both algorithms ended up as local searches around the best point, but with different machinery: the BBOB algorithm uses a GP with expected improvement inside an adaptive trust region (model-based local BO), the real-world algorithm screens random points with a random forest and spends most evaluations on fixed-size Gaussian perturbations of the incumbent (closer to a model-assisted evolution strategy). The key steps towards both designs came from "replace component" mutations (section 5.3).

## 10. Results

All numbers below are produced by `stage2_benchmark.py` and `run_analysis_plots.py` (`outputs/stage2_benchmark/stage2_final_performance.csv`, `outputs/convergence_analysis/stage2_statistics.csv`). 4 methods × 2 problems × 10 seeds, 600 evaluations each, all 80 runs used the full budget (`Run_Status = ok`).

### 10.1 Convergence plots

Best-so-far value, mean ± standard error over the 10 runs, linear axes. The left panel shows the whole run; because the first evaluations are orders of magnitude larger than the rest, the right panel zooms in on evaluations 60 (10·d, the stage 1 budget) to 600.

**BBOB F8 (loss $f - f_{opt}$)**

![BBOB](outputs/convergence_analysis/stage2_bbob_loss_convergence.png)

**Real-world 1124 (objective $f$)**

![RW](outputs/convergence_analysis/stage2_realworld_objective_convergence.png)

Stage 1 search progress: `outputs/convergence_analysis/stage1_aocc_history.png` and the LLaMEA-BO plots in `outputs/convergence_analysis/stage1_*/`.

### 10.2 Final value after 600 evaluations and AOCC

Final value: mean ± standard error over 10 runs and median (lower is better). AOCC of the whole run of 600 evaluations, computed as in stage 1: mean ± std and median (higher is better). p: two-sided Wilcoxon rank-sum test against the best method of that column (best median), Holm corrected per problem.

| Problem | Method | Final: mean ± SE | Final: median | p (final) | AOCC: mean ± std | AOCC: median | p (AOCC) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BBOB F8 (loss $f - f_{opt}$) | **Generated on BBOB (`AdaptiveTrustRegionBO`)** | 11.0 ± 6.2 | **3.35** | — | **0.473 ± 0.053** | **0.476** | — |
| | CMA-ES | **4.20 ± 0.32** | 4.39 | 0.43 | 0.429 ± 0.019 | 0.436 | 0.063 |
| | Generated on real-world (`EMBO_SB`) | 57.2 ± 20.6 | 11.2 | 0.022 | 0.422 ± 0.028 | 0.428 | 0.063 |
| | BoTorch vanilla BO | 35.2 ± 5.1 | 37.0 | 0.022 | 0.414 ± 0.011 | 0.416 | 0.063 |
| Real-world 1124 (objective $f$) | **CMA-ES** | **1.84 ± 0.17** | **1.92** | — | 0.501 ± 0.008 | 0.501 | 0.18 |
| | Generated on real-world (`EMBO_SB`) | 1.98 ± 0.11 | 2.00 | 0.57 | **0.505 ± 0.004** | **0.504** | — |
| | BoTorch vanilla BO | 2.32 ± 0.15 | 2.29 | 0.18 | 0.502 ± 0.003 | 0.502 | 0.18 |
| | Generated on BBOB (`AdaptiveTrustRegionBO`) | 2.87 ± 0.19 | 2.97 | 0.011 | 0.499 ± 0.006 | 0.498 | 0.11 |

Median best-so-far at intermediate budgets (loss for F8, objective for 1124):

| Problem | Method | 60 evals | 150 | 300 | 600 |
|:---|:---|:---:|:---:|:---:|:---:|
| BBOB F8 | Generated on BBOB | **77.4** | **11.1** | **4.37** | **3.35** |
| | Generated on real-world | 893 | 133 | 19.5 | 11.2 |
| | BoTorch | 319 | 130 | 53.0 | 37.0 |
| | CMA-ES | 1021 | 230 | 20.0 | 4.39 |
| Real-world 1124 | Generated on BBOB | **3.33** | 2.97 | 2.97 | 2.97 |
| | Generated on real-world | 3.43 | **2.59** | 2.34 | 2.00 |
| | BoTorch | 3.66 | 3.41 | 2.71 | 2.29 |
| | CMA-ES | 4.50 | 3.80 | **2.21** | **1.92** |

Mean wall clock time per run (one CPU core): `AdaptiveTrustRegionBO` 27–29 min, BoTorch 20–22 min, `EMBO_SB` 4 min, CMA-ES < 1 s.

### 10.3 Discussion

**Specialization of the BBOB algorithm.** On its own target `AdaptiveTrustRegionBO` is the best method at every budget: median loss 77 after 60 evaluations (BoTorch 319, CMA-ES 1021), 11 after 150 and 3.35 after 600, the best final median of all methods, also below CMA-ES (4.39), and the best AOCC (0.476). Its trust region follows the curved Rosenbrock valley, and the GP with expected improvement inside the region refines it efficiently. The difference with CMA-ES is not significant (p = 0.43 on the final value, p = 0.063 on the AOCC), because two runs got stuck (losses 61 and 29 for seeds 117 and 113): the "restart" of the trust region stays around the same best point, so a run that converges to a bad part of the valley cannot leave it. This shows in the large mean (11.0) and in the spread of the AOCC.

**Specialization of the real-world algorithm.** From evaluation 60 on, `EMBO_SB` is better than BoTorch at every budget in the table, from ≈ 75 evaluations on it is better than the BBOB algorithm, and it has the best AOCC on 1124 (0.504), although the AOCC differences on this problem are small and not significant (p ≥ 0.11). CMA-ES catches up after ≈ 260 evaluations and has the best final median (1.92 vs 2.00, p = 0.57). The random forest screening and the Gaussian perturbations of the best point make steady local progress on the rugged FM problem at a low cost per iteration.

**Generalization.** The real-world algorithm transfers well to F8: its local Gaussian search also follows the Rosenbrock valley (median loss 11.2, significantly better than BoTorch with p = 0.022, and after 300 evaluations at the level of CMA-ES), although it is slower than the BBOB algorithm. The BBOB algorithm does not transfer to the real-world problem: after ≈ 150 evaluations it makes no more progress (median 2.97 from evaluation 150 to 600) and with the full budget it is significantly the worst method (p = 0.011). On the multimodal FM problem the trust region shrinks around the first local optimum it finds, and because its "restart" stays around the same point it cannot escape. So both algorithms are local searches, but the one that keeps a fixed step size (`EMBO_SB`) generalizes, while the one whose region collapses around the incumbent is specialised to the single valley of F8.

**Final value vs AOCC.** The two views mostly agree. Differences come from the speed of convergence: on 1124 CMA-ES has the best final value but a lower AOCC than `EMBO_SB` and BoTorch, because it is the slowest method in the first ≈ 200 evaluations; on F8 the BBOB algorithm leads in both. The AOCC (log scale over the whole run) is what stage 1 selected for.

**Baselines.** CMA-ES has the best final value on 1124 and the second best on F8, while the BO methods are better at small budgets. The early lead of BoTorch over CMA-ES on problem 1124 (objective ≈ 3.7 after 12 evaluations in every run) comes from its initial design, not from the model: the provided baseline uses a *centred* Latin hypercube (`criterion="center"`), so every design contains the grid value $x_1 = 0.417$, which corresponds to an almost zero amplitude of the generated sound wave and gives $f \approx 4$ (the "silent signal" plateau) whatever the other variables are. Uniformly random points reach $f < 4.3$ with probability ≈ 0.4 %.

**Limitations.** One search was run per target, so the variability of the LLaMEA search itself is not measured. The stage 1 fitness (60 evaluations) favours algorithms that start fast, and it cannot detect behaviour that only appears later, such as runs that get stuck. Both best algorithms contain parts that do not do what the code claims (a single length scale called "ARD", an LCB without real uncertainty, a sign error without effect), which shows that the AOCC selects behaviour, not the intended design.

## 11. Search behaviour

The best AOCC stops improving after a few generations (generation 2 on BBOB, generation 5 on the real-world problem). The logs suggest three reasons:

* **Most mutations make an algorithm worse.** Only 15 % (BBOB) and 22 % (real-world) of the offspring have a higher AOCC than their parent, partly because many offspring fail, so with elitist (4 + 12) selection the same parents survive for several generations.
* **The population loses its diversity.** All 12 candidates of the last generation descend from a single algorithm of generation 0 (#1 on BBOB, #2 on the real-world problem), so the search becomes a local search around one lineage, even though the lineage itself changes structure through the "replace component" mutations (section 9.1).
* **The fitness is noisy compared with the improvements.** The AOCC of the best algorithm varies with a standard deviation of 0.020 (BBOB) and 0.015 (real-world) over the 10 seeds, so the mean AOCC of a candidate has a standard error of ≈ 0.005–0.006 and the difference between two candidates ≈ 0.008. A truly better candidate is still selected more often than not, but differences below ≈ 0.01 cannot be told apart reliably. In addition, every candidate is evaluated only once on the same 10 seeds, and the elite are the highest of many noisy scores, so their AOCC is biased upward and is never re-evaluated, which keeps the same parents in the population. The late changes of the best AOCC (e.g. 0.369 → 0.367 → 0.368 on BBOB) are within this noise.

Raising the temperature (0.8 now) would mainly add token-level randomness (more varied details, but also more broken code; syntax errors are already the most frequent failure on BBOB) and does not address the selection, lineage and noise effects above. More effective options would be diversity-preserving selection (more parents, niching by design type, non-elitist selection), more seeds or a larger budget per candidate to reduce the fitness noise, and evaluating candidates at more than one budget.

## 12. External code attribution

* **`llamevol/`, `misc/`, `analysis/`**: adapted from **LLaMEA-BO** by Wenhu Li, Niki van Stein, Thomas Bäck and Elena Raponi, https://github.com/BayesOptApp/LLaMEA-BO/ (based on LLaMEA, https://github.com/XAI-liacs/LLaMEA).
* **`baseline/`, `Practical_Problems/`**: provided by the course instructors (Iván Olarte Rodríguez, Elena Raponi, Santiago García Mayayo), https://github.com/BayesOptApp/BO_Course_Assignment.
* **CMA-ES**: pycma, https://github.com/CMA-ES/pycma.
