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
- SobolMaternEIBO: 0.4748, 32.63 seconds, The algorithm is named **SobolMaternEIBO** and combines several innovative components for efficient Bayesian Optimization. It uses a Sobol sequence for initial sampling and candidate generation to ensure space-filling properties. The surrogate model employs a Matern kernel with ARD to adaptively learn feature relevance, and Expected Improvement (EI) is used as the acquisition function with a dynamic exploration strategy. For acquisition function optimization, the algorithm generates 50 Sobol candidates and selects the best, balancing exploration and computational efficiency. The trust region approach is implicitly addressed via the Matern kernel's local modeling properties.


- SobolMaternEIBO: 0.4656, 2.64 seconds, This Bayesian Optimization (BO) algorithm, named **SobolMaternEIBO**, combines a space-filling initial design, a computationally efficient Gaussian Process (GP) with a Matern kernel, and a batch Expected Improvement (EI) acquisition function. The algorithm is optimized for parallel evaluations, computational efficiency, and adaptability to high-dimensional problems. Key components include:  
1. **Initial Design**: Sobol sampling ensures uniform coverage of the search space.  
2. **Surrogate Model**: A GP with a Matern-1.5 kernel and ARD length scales, optimized for moderate computational cost and flexibility.  
3. **Acquisition Function**: Expected Improvement (EI) balances exploration and exploitation, with batch selection via random candidate sampling.  
4. **Batch Selection**: Efficiently selects diverse candidates by evaluating EI on 200 randomly sampled points and selecting the top batch.  
5. **Computation Control**: Uses minimal hyperparameter restarts (0) and smaller batches to avoid expensive GP refits.  


- SobolMaternEIBO: 0.0000, 0.00 seconds, SobolMaternEIBO is a Bayesian Optimization algorithm with the following components:  
1. **Sobol Sequences** for initial sampling to ensure space-filling coverage.  
2. **Gaussian Process (GP) Surrogate** with anisotropic (ARD) Matern kernel (nu=2.5) to model the objective function.  
3. **Expected Improvement (EI)** acquisition function for balancing exploration and exploitation.  
4. **L-BFGS-B Optimization** of the acquisition function, initialized from top candidates sampled from the search space.  
The algorithm is designed for moderate-dimensional (e.g., 6D) problems with a computationally efficient per-iteration cost, suitable for parallel evaluation constraints.


- LogEIAdaptBO: 0.0000, 0.00 seconds, LogEIAdaptBO is a Bayesian Optimization algorithm designed for efficient global optimization. It combines the strengths of low-discrepancy sampling, adaptive Gaussian Process modeling, and log-transformed Expected Improvement for acquisition. The algorithm uses Sobol sequences for initial exploration, an anisotropic Matérn kernel for the GP to capture complex interactions between dimensions, and LogEI to balance exploration and exploitation while simplifying the acquisition function landscape for optimization.




The selected solution to update is:
LogEIAdaptBO is a Bayesian Optimization algorithm designed for efficient global optimization. It combines the strengths of low-discrepancy sampling, adaptive Gaussian Process modeling, and log-transformed Expected Improvement for acquisition. The algorithm uses Sobol sequences for initial exploration, an anisotropic Matérn kernel for the GP to capture complex interactions between dimensions, and LogEI to balance exploration and exploitation while simplifying the acquisition function landscape for optimization.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class LogEIAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        # Create Matérn kernel with ARD (anisotropic)
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2),
                              ard_kernel=True)
        # Fit Gaussian Process
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        best_y = np.min(self.y)
        imp = best_y - mean
        imp = np.where(imp < 0, 0, imp)
        Z = imp / (std + 1e-8)
        # Expected improvement
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        # LogEI for optimization
        log_ei = np.log(ei + 1e-10)
        return log_ei

    def _select_next_points(self, batch_size):
        # Generate candidate points
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=10)
        candidates = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidates

        # Evaluate acquisition function for candidates
        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmax(acq_vals)
        best_candidate = candidates[best_idx]

        # Optimize the acquisition function
        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],  # Maximize by minimizing negative
            x0=best_candidate,
            bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        return result.x[np.newaxis, :]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        while self.n_evals < self.budget:
            # Fit model
            self.gp = self._fit_model(self.X, self.y)

            # Select next point
            next_point = self._select_next_points(1)

            # Evaluate
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]

```
An error occurred : Traceback (most recent call last):
  File "<LogEIAdaptBO>", line 94, in __call__
  94->             self.gp = self._fit_model(self.X, self.y)
  File "<LogEIAdaptBO>", line 28, in _fit_model
  26 |     def _fit_model(self, X, y):
  27 |         # Create Matérn kernel with ARD (anisotropic)
  28->         kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
  29 |                               length_scale_bounds=(1e-2, 1e2),
  30 |                               ard_kernel=True)
TypeError: Matern.__init__() got an unexpected keyword argument 'ard_kernel'


Replace the exploration strategy of the selected solution by a substantially different alternative (options: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points), and adapt the rest of the algorithm where needed. Explain in the justification why the new exploration strategy should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

