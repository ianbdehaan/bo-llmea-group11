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
TrustRegionEI_BO is a Bayesian Optimization algorithm tailored for the Rosenbrock function, combining trust region exploration, Expected Improvement (EI) acquisition, and efficient Gaussian Process (GP) modeling. The algorithm uses a Sobol sequence for initial sampling, a GP with anisotropic Matern kernel, and dynamically adjusts the search region (trust region) around the current best point. The trust region expands/shrinks based on evaluation improvements, balancing exploration and exploitation for the narrow valley structure of Rosenbrock.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
import numpy as np
import warnings


class TrustRegionEI_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 2  # Initial sample size
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.best_x = None
        self.best_y = np.inf
        self.trust_radius = 0.5  # Initial trust region radius

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        # Anisotropic Matern kernel with ARD
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds="fixed"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=3,
                random_state=int(self.rng.integers(2**31 - 1)),
            )
            gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu - self.best_y
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _select_next_points(self, model, batch_size=1):
        # Generate candidates in trust region
        candidates = self.rng.uniform(
            low=self.best_x - self.trust_radius,
            high=self.best_x + self.trust_radius,
            size=(10 * self.dim, self.dim),
        )
        # Clip to bounds
        candidates = np.clip(candidates, self.bounds[0], self.bounds[1])

        # Evaluate EI and optimize
        ei = self._acquisition_function(candidates, model)
        best_idx = np.argmax(ei)
        x0 = candidates[best_idx]

        # Refine with L-BFGS-B
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1), model).ravel()[0]

        bounds = [(l, u) for l, u in zip(self.best_x - self.trust_radius, self.best_x + self.trust_radius)]
        res = qmc.optimize(
            objective,
            bounds=bounds,
            method="L-BFGS-B",
            x0=x0,
            options={"maxiter": 50},
        )
        return res.x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        for i in range(len(X)):
            if y[i] < self.best_y:
                self.best_y = y[i]
                self.best_x = X[i].copy()
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.n_evals += self.n_init

        # Build initial model
        model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Select next point
            next_X = self._select_next_points(model)
            y_new = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, y_new)
            self.n_evals += len(next_X)

            # Update model and trust region
            model = self._fit_model(self.X, self.y)
            # Adjust trust region
            improvement = self.best_y - y_new[0]
            if improvement > 1e-4:
                self.trust_radius = min(1.2 * self.trust_radius, 2.0)
            else:
                self.trust_radius = max(0.5 * self.trust_radius, 0.1)

        return self.best_y, self.best_x

```
An error occurred : Traceback (most recent call last):
  File "<TrustRegionEI_BO>", line 112, in __call__
 112->             next_X = self._select_next_points(model)
  File "<TrustRegionEI_BO>", line 66, in _select_next_points
  66->         ei = self._acquisition_function(candidates, model)
  File "<TrustRegionEI_BO>", line 51, in _acquisition_function
  49 |             imp = mu - self.best_y
  50 |             Z = imp / sigma
  51->             ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
  52 |             ei[sigma == 0.0] = 0.0
  53 |         return ei
NameError: name 'norm' is not defined


Replace the exploration strategy of the selected solution by a substantially different alternative (options: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points), and adapt the rest of the algorithm where needed. Explain in the justification why the new exploration strategy should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

