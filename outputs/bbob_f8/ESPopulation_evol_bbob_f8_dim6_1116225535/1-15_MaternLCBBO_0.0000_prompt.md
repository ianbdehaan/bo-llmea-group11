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
This algorithm combines a Matérn-ARD kernel Gaussian Process (GP) as a surrogate with a decaying Lower Confidence Bound (LCB) acquisition function, optimized using random restarts and L-BFGS-B. It uses a Sobol sequence for initial sampling and dynamically adjusts exploration based on the number of iterations to balance global and local search. The key features are:  
- **Sobol initial design** for space-filling sampling  
- **Matérn-ARD kernel GP** with automatic relevance determination  
- **Decaying LCB acquisition function**: exploration weight `κ = log(1 + t)/t` where `t` is the iteration count  
- **Randomized optimization** of the acquisition function (50 random starts + L-BFGS-B)  
- **Dynamic exploration/exploitation balance** via time-dependent `κ`  


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import numpy as np
import warnings

class MaternLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = dim + 1  # Initial design size
        self.rng = np.random.default_rng(42)  # Fixed seed for deterministic runs

    def _sample_points(self, n_points):
        """Generate Sobol samples for initial design"""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        return samples * 10.0 - 5.0  # Scale from [0,1]^d to [-5,5]^d

    def _fit_model(self, X, y):
        """Fit a GP with Matérn-ARD kernel"""
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore GP warnings during fit
            gp = GaussianProcessRegressor(
                kernel=kernel,
                optimizer='fmin_l_bfgs_b',
                n_restarts_optimizer=0,  # Minimal restarts for speed
                alpha=1e-6,  # Small noise variance
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        """Decaying LCB acquisition function"""
        mu, sigma = self.model.predict(X, return_std=True)
        t = self.n_evals - self.n_init  # Iteration counter after initial samples
        if t <= 0:
            # Early exploration phase
            kappa = 2.0
        else:
            # Decay exploration over time
            kappa = np.log(1 + t) / t
        return mu - kappa * sigma

    def _select_next_points(self, batch_size):
        """Optimize acquisition function with random restarts"""
        n_rand = 50  # Number of random candidates
        X_rand = self._sample_points(n_rand)
        # Optimize each candidate with L-BFGS-B
        best_vals = []
        best_points = []
        for x0 in X_rand:
            res = self._optimize_acq(x0)
            if res.success:
                best_vals.append(-res.fun)  # Negative because we're maximizing
                best_points.append(res.x)
        
        if not best_points:
            # Fallback to random sampling if no optimization succeeded
            return self._sample_points(batch_size)
        
        # Return the point with best acquisition value
        best_idx = np.argmax(best_vals)
        return np.array([best_points[best_idx]])

    def _optimize_acq(self, x0):
        """Optimize acquisition function from a starting point"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore scipy optimization warnings
            result = self.model.optimizer(
                x0=x0,
                bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            )
        return result

    def _evaluate_points(self, func, X):
        """Evaluate points with function"""
        n_points = len(X)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update history of evaluated points"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """Main optimization loop"""
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            # Select next point to evaluate
            next_point = self._select_next_points(1)
            # Evaluate next point
            next_y = self._evaluate_points(func, next_point)
            # Update history
            self._update_eval_points(next_point, next_y)
            # Update model
            self.model = self._fit_model(self.X, self.y)
            # Update best
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx] < best_y:
                best_y = self.y[current_best_idx]
                best_x = self.X[current_best_idx]

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<MaternLCBBO>", line 115, in __call__
 115->             next_point = self._select_next_points(1)
  File "<MaternLCBBO>", line 61, in _select_next_points
  61->             res = self._optimize_acq(x0)
  File "<MaternLCBBO>", line 78, in _optimize_acq
  76 |         with warnings.catch_warnings():
  77 |             warnings.simplefilter("ignore")  # Ignore scipy optimization warnings
  78->             result = self.model.optimizer(
  79 |                 x0=x0,
  80 |                 bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
TypeError: 'str' object is not callable


Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

