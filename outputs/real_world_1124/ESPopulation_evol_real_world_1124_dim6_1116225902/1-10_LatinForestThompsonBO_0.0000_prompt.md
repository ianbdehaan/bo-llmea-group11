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
SobolMaternEIBO is a Bayesian Optimization algorithm with the following components:  
1. **Sobol Sequences** for initial sampling to ensure space-filling coverage.  
2. **Gaussian Process (GP) Surrogate** with anisotropic (ARD) Matern kernel (nu=2.5) to model the objective function.  
3. **Expected Improvement (EI)** acquisition function for balancing exploration and exploitation.  
4. **L-BFGS-B Optimization** of the acquisition function, initialized from top candidates sampled from the search space.  
The algorithm is designed for moderate-dimensional (e.g., 6D) problems with a computationally efficient per-iteration cost, suitable for parallel evaluation constraints.


With code:
```python
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from collections.abc import Callable

class SobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 2 * dim)  # Ensure sufficient initial samples
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))  # Seeded RNG

    def _sample_points(self, n_points):
        # Generate quasi-random Sobol samples
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return scaled_samples

    def _fit_model(self, X, y):
        # GP with Matern kernel (ARD)
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement for minimization
        mu, sigma = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mu
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + sigma * norm.pdf(z)
        ei[sigma == 0.] = max(best_y - mu[sigma == 0.], 0.0)
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Optimize EI using random sampling and gradient-based refinement
        n_candidates = 50
        candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(candidates)
        idx = np.argsort(acq_values, axis=0)[::-1]
        top_candidates = candidates[idx[:10]]  # Top 10 candidates for L-BFGS-B
        res = []
        for x0 in top_candidates:
            x0 = x0.reshape(1, -1)
            result = minimize(
                lambda x: -self._acquisition_function(x.reshape(1, -1))[0],
                x0=x0,
                bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            res.append((result.x, -result.fun))
        best_x, best_acq = max(res, key=lambda x: x[1])
        return best_x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate func for each row in X
        y = np.array([func(x) for x in X])
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with Sobol samples
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)
        self.n_evals = len(self.X)
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            current_y = new_y[0][0]
            if current_y < best_y:
                best_y = current_y
                best_x = next_point[0]

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<SobolMaternEIBO>", line 94, in __call__
  94->             next_point = self._select_next_points(1)
  File "<SobolMaternEIBO>", line 52, in _select_next_points
  52->         acq_values = self._acquisition_function(candidates)
  File "<SobolMaternEIBO>", line 45, in _acquisition_function
  43 |             z = improvement / sigma
  44 |             ei = (improvement * norm.cdf(z)) + sigma * norm.pdf(z)
  45->         ei[sigma == 0.] = max(best_y - mu[sigma == 0.], 0.0)
  46 |         return ei.reshape(-1, 1)
  47 | 
ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.


Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

