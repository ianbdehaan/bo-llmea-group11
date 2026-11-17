You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on the real-world problem 1124 (parameter estimation of Frequency Modulated Sound Waves), dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.
A BO algorithm can be built from many different components, for example:
- initial design: Sobol, Latin hypercube, random, maximin designs, different sizes (e.g. d+1 up to 4d points)
- surrogate model: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y, random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted only near the best point
- acquisition function: expected improvement, log expected improvement, lower confidence bound with a schedule for the exploration weight, probability of improvement, Thompson sampling from the GP posterior, a portfolio that switches between several acquisition functions
- acquisition function optimizer: L-BFGS-B from several starting points, differential evolution, a small evolution strategy, random candidates plus local refinement, candidates sampled around the best points
- exploration strategy: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points
Do not always use the same default choices (e.g. a scikit-learn GP with expected improvement on random candidates); choose the components that suit the problem.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ThompsonSamplingEnsembleMaternBO: 0.4773, 10.68 seconds, **ThompsonSamplingEnsembleMaternBO** combines several improvements to the original ThompsonSamplingMaternBO algorithm. The algorithm uses an expanded initial design (2d+1 Sobol points) to better inform the surrogate model early on. It generates 200 Sobol candidates per iteration for Thompson Sampling, increasing exploration breadth. The batch size is increased to 2, allowing parallel evaluation of two points in each iteration to reduce model refits. The GP hyperparameters are refit less frequently (every 5 iterations) to balance model accuracy and computational cost. A Matérn-5/2 kernel with ARD is used for anisotropic modeling, and a log-transformed objective function improves stability in the GP fit.


- TrustRegionSobolMaternEIBO: 0.4765, 214.86 seconds, **TrustRegionSobolMaternEIBO** combines a dynamic trust region strategy with Sobol-based candidate generation and Expected Improvement (EI) as the acquisition function. The algorithm starts with a Sobol sequence to initialize the search space, ensuring diverse sampling. A Gaussian Process with a Matérn kernel models the objective function, and the trust region adaptively expands or contracts based on evaluation progress. The EI acquisition function optimizes exploration-exploitation balance by prioritizing candidates that offer the highest expected improvement over the current best solution.  


- TrustRegionSobolMaternEIBO: 0.4762, 38.72 seconds, **TrustRegionSobolMaternEIBO** replaces the global Sobol-based exploration in SobolMaternEIBO with a dynamic trust region strategy inspired by TuRBO. The algorithm maintains a trust region centered on the best solution found so far, adjusting its size based on evaluation progress. Key components:  
1. **Sobol Initial Design** for diverse initial sampling.  
2. **Gaussian Process Surrogate** with anisotropic Matérn kernel.  
3. **Expected Improvement Acquisition Function** with constrained optimization within the trust region.  
4. **Dynamic Trust Region**: Expands when progress is made (indicating a promising area), contracts when progress stalls (avoiding premature convergence).  
5. **Candidate Sampling**: Sobol candidates are generated within the current trust region, ensuring focused local search while preserving diversity.  

The trust region strategy balances global and local exploration by adaptively adjusting the search space, reducing redundant evaluations and improving convergence efficiency.  


- RefinedTrustRegionSobolMaternEIBO: 0.4758, 7.45 seconds, **RefinedTrustRegionSobolMaternEIBO** improves upon TrustRegionSobolMaternEIBO by optimizing trust region dynamics, candidate diversity, and model efficiency. Key refinements include:
1. **Adaptive Trust Region**: Uses smaller initial radius (1.0) and smoother adjustments (expansion/contraction by 1.1/0.8) to avoid overexploration.  
2. **Enhanced Candidate Diversity**: Generates 100 Sobol candidates per iteration with batch selection (2 points) to balance exploration and parallelism.  
3. **Lightweight Model Refitting**: Reduces GP hyperparameter restarts to 1 and refits less frequently (every 2 iterations) when progress stalls.  
4. **Dynamic Radius Bounds**: Trust region bounds are clamped dynamically to prevent boundary artifacts while preserving diversity.  

The algorithm maintains a focus on efficient exploration while reducing model overhead, improving convergence on high-dimensional FM wave parameter estimation.




The selected solution to update is:
**ThompsonSamplingEnsembleMaternBO** combines several improvements to the original ThompsonSamplingMaternBO algorithm. The algorithm uses an expanded initial design (2d+1 Sobol points) to better inform the surrogate model early on. It generates 200 Sobol candidates per iteration for Thompson Sampling, increasing exploration breadth. The batch size is increased to 2, allowing parallel evaluation of two points in each iteration to reduce model refits. The GP hyperparameters are refit less frequently (every 5 iterations) to balance model accuracy and computational cost. A Matérn-5/2 kernel with ARD is used for anisotropic modeling, and a log-transformed objective function improves stability in the GP fit.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class ThompsonSamplingEnsembleMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Expanded initial design
        self.refit_interval = 5  # Model refits every 5 iterations
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None

    def _sample_points(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Log-transform objective values for numerical stability
        y_log = np.log(y + 1e-8)  # Small epsilon to avoid log(0)
        
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,  # Fast refits with no restarts
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_log)
        return model

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(200)  # 200 Sobol candidates for TS
        sampled_functions = self.model.sample_y(candidates, n_samples=1)
        best_idx = np.argmin(sampled_functions)
        next_X = candidates[best_idx].reshape(1, -1)
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)  # Refit periodically
            
            batch_size = 2  # Evaluate two points in parallel
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm ThompsonSamplingEnsembleMaternBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4773 with standard deviation 0.0079.

took 10.68 seconds to run.

Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

