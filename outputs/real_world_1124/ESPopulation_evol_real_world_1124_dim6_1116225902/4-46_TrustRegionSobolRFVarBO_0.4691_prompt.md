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
**TrustRegionSobolMaternEIBO** replaces the global Sobol-based exploration in SobolMaternEIBO with a dynamic trust region strategy inspired by TuRBO. The algorithm maintains a trust region centered on the best solution found so far, adjusting its size based on evaluation progress. Key components:  
1. **Sobol Initial Design** for diverse initial sampling.  
2. **Gaussian Process Surrogate** with anisotropic Matérn kernel.  
3. **Expected Improvement Acquisition Function** with constrained optimization within the trust region.  
4. **Dynamic Trust Region**: Expands when progress is made (indicating a promising area), contracts when progress stalls (avoiding premature convergence).  
5. **Candidate Sampling**: Sobol candidates are generated within the current trust region, ensuring focused local search while preserving diversity.  

The trust region strategy balances global and local exploration by adaptively adjusting the search space, reducing redundant evaluations and improving convergence efficiency.  


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionSobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)  # Ensure at least 5 initial points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        # Trust region parameters
        self.tr_center = None
        self.tr_radius = 2.5  # Initial radius (25% of domain range)
        self.max_tr_radius = 5.0  # Half of domain range
        self.min_tr_radius = 0.1  # Minimum allowable radius

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Sobol candidates within specified bounds (default: overall domain)."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Gaussian Process with Matérn kernel and ARD."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,  # Reduced restarts for speed
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle zero variance
        return ei

    def _select_next_points(self):
        """Generate candidates within current trust region."""
        if self.tr_center is None:
            return self._sample_points(50)  # Fallback to global search if no trust region
        
        # Compute trust region bounds (clamped to domain)
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        
        candidates = self._sample_points(50, low=low, high=high)  # 50 candidates in trust region
        acq_vals = self._acquisition_function(candidates)
        best_acq_idx = np.argmax(acq_vals)
        next_X = candidates[best_acq_idx].reshape(1, -1)
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Sobol
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Set initial trust region center
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update trust region
            current_best_idx = np.argmin(self.y)
            improved = next_y[0] < self.y[best_idx]
            
            if improved:
                # Expand trust region when progress is made
                self.tr_radius = min(self.tr_radius * 1.2, self.max_tr_radius)
            else:
                # Contract when no improvement
                self.tr_radius = max(self.tr_radius * 0.9, self.min_tr_radius)
            
            # Update best solution and trust region center
            best_idx = current_best_idx
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm TrustRegionSobolMaternEIBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4762 with standard deviation 0.0083.

took 38.72 seconds to run.

Replace the surrogate model of the selected solution by a substantially different alternative (options: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y, random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted only near the best point), and adapt the rest of the algorithm where needed. Explain in the justification why the new surrogate model should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

