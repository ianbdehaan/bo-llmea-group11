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
- EnsembleEIBO: 0.3044, 44.76 seconds, EnsembleEIBO is a Bayesian Optimization algorithm that leverages an ensemble of Random Forests as the surrogate model and Expected Improvement (EI) acquisition function. The algorithm dynamically shifts from global to local exploration by using a hybrid acquisition optimization strategy: (1) generating diverse candidates via Sobol sampling, (2) refining these candidates with L-BFGS-B, and (3) prioritizing regions near the best solution found so far. Key innovations include:  
- **Random Forest Ensemble**: Uncertainty estimation via prediction variance across trees, avoiding GP hyperparameter optimization.  
- **Dynamic Candidate Refinement**: Combines global Sobol candidates with local optimization and prioritizes regions near the best solution.  
- **Budget-Aware Initialization**: Reduces initial design points (2d instead of 3d) to save evaluations for later iterations.  


- ThompsonSamplingBO: 0.2943, 146.06 seconds, ThompsonSamplingBO is a Bayesian Optimization algorithm that utilizes Thompson Sampling for acquisition function optimization. It combines an anisotropic Matérn kernel Gaussian Process (GP) as a surrogate model with a dynamic sampling strategy from the posterior. The algorithm employs Sobol sequences for initial space-filling sampling and optimizes sampled functions from the GP posterior to balance exploration and exploitation. By sampling a function realization from the GP and optimizing it, the algorithm adapts to the Rosenbrock function's elongated valley structure, promoting efficient exploration while maintaining computational efficiency through random restarts and L-BFGS-B local optimization.


- DynamicLCBBO: 0.2943, 167.75 seconds, DynamicLCBBO is a Bayesian Optimization algorithm that uses a dynamic Lower Confidence Bound (LCB) acquisition function with an exploration coefficient that decays over time. The algorithm combines efficient space coverage through Sobol sequences, a Gaussian Process (GP) with anisotropic Matérn kernel for surrogate modeling, and a hybrid acquisition optimization approach involving random sampling and local search with bounds. The dynamic LCB balances exploration and exploitation by reducing exploration as more evaluations are made, making it well-suited for the Rosenbrock function's elongated valley structure. The algorithm is designed for computational efficiency with limited hyperparameter tuning and careful budget management.


- ForestDynamicLCBBO: 0.2590, 215.80 seconds, ForestDynamicLCBBO is a Bayesian Optimization algorithm that employs a **Random Forest surrogate model** with uncertainty estimation via ensemble variance, a **dynamic Lower Confidence Bound (LCB)** acquisition function, and **Differential Evolution** for global acquisition function optimization. The algorithm uses **Latin Hypercube Sampling** for initial design and adapts exploration dynamically through a decaying coefficient. Key innovations include replacing Gaussian Processes with Random Forest for robustness, using differential evolution for global acquisition function optimization, and avoiding trust regions in favor of global search.




The selected solution to update is:
ForestDynamicLCBBO is a Bayesian Optimization algorithm that employs a **Random Forest surrogate model** with uncertainty estimation via ensemble variance, a **dynamic Lower Confidence Bound (LCB)** acquisition function, and **Differential Evolution** for global acquisition function optimization. The algorithm uses **Latin Hypercube Sampling** for initial design and adapts exploration dynamically through a decaying coefficient. Key innovations include replacing Gaussian Processes with Random Forest for robustness, using differential evolution for global acquisition function optimization, and avoiding trust regions in favor of global search.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress sklearn warnings

class ForestDynamicLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = max(2 * dim + 1, 20)  # Latin Hypercube initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None  # Random Forest surrogate
        self.kappa_decay = 0.95  # Decay rate for exploration coefficient
        self.initial_kappa = 2.0  # Initial exploration weight

    def _sample_points(self, n_points):
        # Latin Hypercube Sampling for space-filling initial design
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Random Forest with uncertainty estimation
        model = RandomForestRegressor(
            n_estimators=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            bootstrap=False,
            n_jobs=-1  # Enable parallel training
        )
        model.fit(X, y.ravel())  # Fit on 1D target
        return model

    def _acquisition_function(self, X):
        # Compute LCB using mean and variance from Random Forest
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(tree_preds, axis=0)
        var = np.var(tree_preds, axis=0)
        best_y = np.min(self.y)
        t = self.n_evals
        kappa = self.initial_kappa * (self.kappa_decay ** t)  # Exponential decay
        lcb = mean - kappa * np.sqrt(var)
        return lcb.reshape(-1, 1)

    def _acquisition_objective(self, x):
        # Wrapper for differential evolution (maximize LCB)
        x = np.array(x).reshape(1, -1)
        return -self._acquisition_function(x)[0, 0]  # Negative for minimization

    def _select_next_point(self):
        # Differential Evolution to maximize acquisition function
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(
            self._acquisition_objective,
            bounds,
            strategy='best1bin',
            maxiter=5,  # Limited iterations to save time
            popsize=10,  # Balanced population size
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.rng
        )
        best_candidate = result.x
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
            new_point = self._select_next_point()
            new_y = self._evaluate_points(func, new_point)
            self._update_eval_points(new_point, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += 1
        
        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()

```
The algorithm ForestDynamicLCBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2590 with standard deviation 0.0173.

took 215.80 seconds to run.

Replace the acquisition function of the selected solution by a substantially different alternative (options: expected improvement, log expected improvement, lower confidence bound with a schedule for the exploration weight, probability of improvement, Thompson sampling from the GP posterior, a portfolio that switches between several acquisition functions), and adapt the rest of the algorithm where needed. Explain in the justification why the new acquisition function should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

