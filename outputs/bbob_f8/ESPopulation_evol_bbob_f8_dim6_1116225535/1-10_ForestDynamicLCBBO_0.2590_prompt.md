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
- DynamicLCBBO: 0.2943, 167.75 seconds, DynamicLCBBO is a Bayesian Optimization algorithm that uses a dynamic Lower Confidence Bound (LCB) acquisition function with an exploration coefficient that decays over time. The algorithm combines efficient space coverage through Sobol sequences, a Gaussian Process (GP) with anisotropic Matérn kernel for surrogate modeling, and a hybrid acquisition optimization approach involving random sampling and local search with bounds. The dynamic LCB balances exploration and exploitation by reducing exploration as more evaluations are made, making it well-suited for the Rosenbrock function's elongated valley structure. The algorithm is designed for computational efficiency with limited hyperparameter tuning and careful budget management.


- MaternLCBBO: 0.0000, 0.00 seconds, This algorithm combines a Matérn-ARD kernel Gaussian Process (GP) as a surrogate with a decaying Lower Confidence Bound (LCB) acquisition function, optimized using random restarts and L-BFGS-B. It uses a Sobol sequence for initial sampling and dynamically adjusts exploration based on the number of iterations to balance global and local search. The key features are:  
- **Sobol initial design** for space-filling sampling  
- **Matérn-ARD kernel GP** with automatic relevance determination  
- **Decaying LCB acquisition function**: exploration weight `κ = log(1 + t)/t` where `t` is the iteration count  
- **Randomized optimization** of the acquisition function (50 random starts + L-BFGS-B)  
- **Dynamic exploration/exploitation balance** via time-dependent `κ`  


- ATRBO: 0.0000, 0.00 seconds, The algorithm ATRBO (Adaptive Trust Region Bayesian Optimization) combines a trust region strategy with a Gaussian Process (GP) surrogate model and Expected Improvement (EI) acquisition function. The trust region dynamically adjusts around the best observed solution to balance exploration and exploitation. Initial design uses Sobol sequences for space-filling sampling, while the GP with an anisotropic Matérn kernel models the function landscape. The acquisition function is optimized using multiple L-BFGS-B restarts within the trust region. The trust region shrinks gradually over iterations to focus on promising areas.


- TrustRegionEI_BO: 0.0000, 0.00 seconds, TrustRegionEI_BO is a Bayesian Optimization algorithm tailored for the Rosenbrock function, combining trust region exploration, Expected Improvement (EI) acquisition, and efficient Gaussian Process (GP) modeling. The algorithm uses a Sobol sequence for initial sampling, a GP with anisotropic Matern kernel, and dynamically adjusts the search region (trust region) around the current best point. The trust region expands/shrinks based on evaluation improvements, balancing exploration and exploitation for the narrow valley structure of Rosenbrock.




The selected solution to update is:
The algorithm ATRBO (Adaptive Trust Region Bayesian Optimization) combines a trust region strategy with a Gaussian Process (GP) surrogate model and Expected Improvement (EI) acquisition function. The trust region dynamically adjusts around the best observed solution to balance exploration and exploitation. Initial design uses Sobol sequences for space-filling sampling, while the GP with an anisotropic Matérn kernel models the function landscape. The acquisition function is optimized using multiple L-BFGS-B restarts within the trust region. The trust region shrinks gradually over iterations to focus on promising areas.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1  # Space-filling initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 1.0  # Initial trust region size
        self.radius_decay = 0.99  # Gradual shrinking factor
        self.model = None  # GP surrogate

    def _sample_points(self, n_points):
        # Generate initial points via Sobol sampling
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # GP with Matérn kernel for smoothness modeling
        kernel = C(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small nugget for stability
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0  # Reduce computational cost
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI)
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mean - best_y) / std
            ei = (mean - best_y) * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Trust region around current best solution
        current_best_idx = np.argmin(self.y)
        current_best = self.X[current_best_idx]
        lower = np.maximum(current_best - self.trust_radius, self.bounds[0])
        upper = np.minimum(current_best + self.trust_radius, self.bounds[1])
        
        # Generate diverse candidates within trust region
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        candidates = qmc.scale(sampler.random(20), lower, upper - lower)
        
        best_candidate = None
        best_ei = -np.inf
        
        # Optimization of acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]
        
        bounds = [(l, u) for l, u in zip(lower, upper)]
        for x0 in candidates:
            res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.success and res.fun > best_ei:
                best_ei = res.fun
                best_candidate = res.x
        
        self.trust_radius *= self.radius_decay  # Adaptive shrinking
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate candidates without exceeding budget
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        # Maintain evaluated data
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.n_evals += initial_X.shape[0]
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            batch_size = 1  # Sequential optimization
            new_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, new_points)
            self._update_eval_points(new_points, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += batch_size
        
        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()

```
An error occurred : Traceback (most recent call last):
  File "<ATRBO>", line 107, in __call__
 107->             new_points = self._select_next_points(batch_size)
  File "<ATRBO>", line 61, in _select_next_points
  59 |         # Generate diverse candidates within trust region
  60 |         sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
  61->         candidates = qmc.scale(sampler.random(20), lower, upper - lower)
  62 |         
  63 |         best_candidate = None
ValueError: Bounds are not consistent 'l_bounds' < 'u_bounds'


Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

