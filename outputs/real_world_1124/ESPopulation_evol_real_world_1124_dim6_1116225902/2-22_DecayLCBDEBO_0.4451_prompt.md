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


- ThompsonSamplingMaternBO: 0.4732, 17.76 seconds, **ThompsonSamplingMaternBO** is a Bayesian Optimization algorithm that replaces Expected Improvement (EI) with Thompson Sampling (TS) to balance exploration and exploitation. It uses Sobol sequences for initial sampling and candidate generation, a Gaussian Process (GP) with Matern-2.5 kernel for modeling, and evaluates TS candidates over 100 Sobol points. The algorithm reduces initial sampling to $d+1$ points to allocate more evaluations to the TS phase and avoids hyperparameter retraining for efficiency.

---


- TrustRegionLogEIAdaptBO: 0.4730, 138.83 seconds, **TrustRegionLogEIAdaptBO** is a Bayesian Optimization (BO) algorithm that combines log-transformed Expected Improvement (LogEI) acquisition with a dynamic trust region strategy inspired by TuRBO. The algorithm uses a trust region that expands or contracts based on evaluation progress, ensuring focused exploration around promising regions. Key components include:  
1. **Sobol Initial Design**: Provides space-filling initial exploration.  
2. **Gaussian Process (GP) Surrogate**: Anisotropic Matérn-5/2 kernel for adaptive modeling.  
3. **LogEI Acquisition Function**: Stabilizes optimization of the acquisition surface.  
4. **Trust Region Strategy**: Dynamically adjusts the search region based on evaluation success, balancing local exploitation and global exploration.  


- TrustRegionAdaptBO: 0.4697, 141.22 seconds, TrustRegionAdaptBO is a Bayesian Optimization algorithm that integrates a trust region strategy inspired by TuRBO for efficient exploration-exploitation. Key components include:  
1. **Initialization**: Uses Sobol sequences to generate a diverse initial design across the search space.  
2. **Surrogate Model**: Anisotropic Matérn kernel Gaussian Process with ARD length scales to adaptively model the objective function.  
3. **Acquisition Function**: Log-transformed Expected Improvement (LogEI) simplifies optimization of the acquisition function.  
4. **Trust Region Strategy**: Maintains a dynamic trust region around the best point found so far. The trust region expands when progress is observed and contracts when progress stalls, ensuring focused exploration of promising areas.  
5. **Candidate Generation**: Candidates are sampled within the current trust region using Sobol sequences, and acquisition function optimization is constrained to this region.  
The algorithm balances global search and local refinement, improving efficiency for moderate-dimensional optimization tasks under computational constraints.




The selected solution to update is:
**TrustRegionLogEIAdaptBO** is a Bayesian Optimization (BO) algorithm that combines log-transformed Expected Improvement (LogEI) acquisition with a dynamic trust region strategy inspired by TuRBO. The algorithm uses a trust region that expands or contracts based on evaluation progress, ensuring focused exploration around promising regions. Key components include:  
1. **Sobol Initial Design**: Provides space-filling initial exploration.  
2. **Gaussian Process (GP) Surrogate**: Anisotropic Matérn-5/2 kernel for adaptive modeling.  
3. **LogEI Acquisition Function**: Stabilizes optimization of the acquisition surface.  
4. **Trust Region Strategy**: Dynamically adjusts the search region based on evaluation success, balancing local exploitation and global exploration.  


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionLogEIAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.trust_width = 1.0
        self.trust_width_factor = 1.05
        self.trust_width_min = 0.1
        self.trust_width_max = 2.0

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
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
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        log_ei = np.log(ei + 1e-10)
        return log_ei

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        lower = best_point - self.trust_width
        upper = best_point + self.trust_width
        lower = np.clip(lower, self.bounds[0], self.bounds[1])
        upper = np.clip(upper, self.bounds[0], self.bounds[1])

        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=10)
        candidates = lower + (upper - lower) * candidates

        acq_vals = self._acquisition_function(candidates)
        best_idx_candidate = np.argmax(acq_vals)
        best_candidate = candidates[best_idx_candidate]

        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],
            x0=best_candidate,
            bounds=[(lower[i], upper[i]) for i in range(self.dim)],
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
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            prev_best = np.min(self.y)
            self._update_eval_points(next_point, new_y)
            new_best = np.min(self.y)
            self.n_evals += 1

            if new_best < prev_best:
                self.trust_width = min(self.trust_width * self.trust_width_factor, self.trust_width_max)
            else:
                self.trust_width = max(self.trust_width / self.trust_width_factor, self.trust_width_min)

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]

```
The algorithm TrustRegionLogEIAdaptBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4730 with standard deviation 0.0103.

took 138.83 seconds to run.

Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

