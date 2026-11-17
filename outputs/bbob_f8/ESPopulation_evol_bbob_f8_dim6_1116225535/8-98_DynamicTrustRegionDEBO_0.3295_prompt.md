You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on BBOB function 8 (Rosenbrock, original), instance 1, dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
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
- AdaptiveMaternLCBGPBO: 0.3412, 562.24 seconds, **AdaptiveMaternLCBGPBO** is a refined Bayesian Optimization algorithm that introduces an anisotropic Matérn kernel for robust modeling of the Rosenbrock function's smoothness and curvature. It employs exponential decay of the LCB exploration parameter for smoother transition from exploration to exploitation. Key improvements include:  
- **Enhanced Kernel**: Anisotropic Matérn (ν=5/2) kernel captures local non-smoothness while maintaining flexibility.  
- **Exponential κ Schedule**: κ = 2.0·e^(-3·progress) ensures rapid early exploration and focused late-phase exploitation.  
- **Hybrid Candidate Pooling**: 200 Sobol candidates with 20 top candidates for L-BFGS-B refinement improves acquisition optimization.  
- **Expanded Initialization**: 3d initial points provide better coverage of the elongated valley.  
- **Robust GP Training**: 3 random restarts for hyperparameter optimization with reduced noise assumption (α=1e-8).  


- ExponentialMaternLCBBO: 0.3412, 553.51 seconds, ExponentialMaternLCBBO is a Bayesian Optimization algorithm that enhances DynamicLCBGPBO with a **Matérn kernel** (ν=5/2) for robust modeling of the Rosenbrock function's non-smooth valley, an **exponential decay schedule** for the LCB acquisition function to rapidly transition from exploration to exploitation, and a **larger initial Sobol design** (3d points) for better coverage. It uses **200 Sobol candidates** and **top 20 refinement** with L-BFGS-B to efficiently optimize the acquisition function, and increases GP hyperparameter training to **3 random restarts** for better model calibration. These improvements enable faster convergence and better navigation of the elongated valley while maintaining computational efficiency.


- EnhancedAnisotropicLCBBO: 0.3392, 513.60 seconds, EnhancedAnisotropicLCBBO is a Bayesian Optimization algorithm that refines DynamicLCBGPBO with key improvements to better navigate the Rosenbrock function's elongated valley. The algorithm introduces an anisotropic Matern kernel (ν=2.5) for more robust modeling of local smoothness, replaces linear κ-decay in the LCB acquisition function with exponential decay for persistent exploration, increases the initial Sobol design size to 3d for better coverage, and employs 200 Sobol candidates with top-20 refinement in acquisition optimization. These changes enhance model adaptability, exploration-exploitation balance, and search efficiency.


- AdaptiveMaternLCBGPBO: 0.3363, 413.88 seconds, AdaptiveMaternLCBGPBO is a refined Bayesian Optimization algorithm that introduces an anisotropic Matérn kernel for robust modeling of the Rosenbrock function's smoothness and curvature. It employs exponential decay of the LCB exploration parameter for smoother transition from exploration to exploitation. Key improvements include:  
- **Enhanced Kernel**: Anisotropic Matérn (ν=5/2) kernel captures local non-smoothness while maintaining flexibility.  
- **Exponential κ Schedule**: κ = e^(-3·progress) ensures rapid early exploration and focused late-phase exploitation.  
- **Hybrid Candidate Pooling**: 150 Sobol candidates with 15 top candidates for L-BFGS-B refinement improves acquisition optimization.  
- **Expanded Initialization**: 3d initial points provide better coverage of the elongated valley.  
- **Robust GP Training**: 3 random restarts for hyperparameter optimization with reduced noise assumption (α=1e-8).  




The selected solution to update is:
EnhancedAnisotropicLCBBO is a Bayesian Optimization algorithm that refines DynamicLCBGPBO with key improvements to better navigate the Rosenbrock function's elongated valley. The algorithm introduces an anisotropic Matern kernel (ν=2.5) for more robust modeling of local smoothness, replaces linear κ-decay in the LCB acquisition function with exponential decay for persistent exploration, increases the initial Sobol design size to 3d for better coverage, and employs 200 Sobol candidates with top-20 refinement in acquisition optimization. These changes enhance model adaptability, exploration-exploitation balance, and search efficiency.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class EnhancedAnisotropicLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Enhanced initial Sobol design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.log2 = math.log(2)

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        current_kappa = self._compute_kappa()
        return mu - current_kappa * sigma  # Minimize LCB

    def _compute_kappa(self):
        progress_ratio = self.n_evals / self.budget
        return 2.0 * np.exp(-progress_ratio * self.log2)  # Exponential decay

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(200)  # Increased Sobol candidates
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)  # Minimize LCB
        top_candidates = candidates[idxs[:20]]  # Top 20 candidates for refinement

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 50}  # Limit iterations to save time
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmin(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        y = np.zeros((n_new, 1))
        for i in range(n_new):
            y[i, 0] = func(X[i])
        self.n_evals += n_new
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial Sobol design for diverse sampling
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize GP model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x

```
The algorithm EnhancedAnisotropicLCBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3392 with standard deviation 0.0199.

took 513.60 seconds to run.

Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

