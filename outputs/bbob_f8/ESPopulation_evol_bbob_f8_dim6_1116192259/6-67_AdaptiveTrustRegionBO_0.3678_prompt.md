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
- AdaptiveTrustRegionBO: 0.3692, 31.23 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- AdaptiveTrustRegionBO: 0.3687, 31.77 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- AdaptiveTrustRegionBO: 0.3674, 20.58 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- AdaptiveTrustRegionBO: 0.3672, 39.48 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.




The selected solution to update is:
**AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class AdaptiveTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # GP model parameters
        self.gp_model = None
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 1.2
        self.progress_threshold = 0.01  # Minimum improvement to consider as progress
        self.stagnation_count = 0
        self.max_stagnation = 5  # Number of iterations without progress before shrinking
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for trust region adaptation
        self.best_y_history = []
        self.last_improvement_iter = 0

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Initialize with a good set of starting points within the trust region
        n_starting_points = min(50 * self.dim, 1000)  # Reduced from previous version for efficiency
        
        # Generate candidates within the trust region around the current best point
        if self.X is not None and len(self.X) > 0:
            # Use the current best point as center of trust region
            best_x = self.X[np.argmin(self.y)]
            
            # Sample points within trust region
            candidates = []
            while len(candidates) < n_starting_points:
                # Generate random points in the trust region
                sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.trust_region_radius
                
                # Ensure points are within bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                candidates.append(sample)
            
            candidates = np.array(candidates)
        else:
            # If no points yet, use Latin hypercube sampling
            candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates (use a simple heuristic - just the best ones)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate all points in X
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y_vals
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        if len(self.best_y_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        # Update stagnation count
        if improvement > self.progress_threshold:
            self.stagnation_count = 0
            # Expand trust region if progress is good
            self.trust_region_radius = min(
                self.trust_region_radius * self.trust_region_expand_factor,
                self.max_trust_region_radius
            )
        else:
            self.stagnation_count += 1
            # Shrink trust region if no significant progress
            if self.stagnation_count >= self.max_stagnation:
                self.trust_region_radius = max(
                    self.trust_region_radius * self.trust_region_shrink_factor,
                    self.min_trust_region_radius
                )
                self.stagnation_count = 0

    def _check_restart_condition(self):
        """Check if we should restart the optimization"""
        # Restart if trust region becomes too small and no progress
        if (self.trust_region_radius <= self.min_trust_region_radius and 
            self.stagnation_count >= self.max_stagnation):
            return True
        return False

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # Check for restart condition
            if self._check_restart_condition():
                # Reset trust region and continue
                self.trust_region_radius = 1.0
                self.stagnation_count = 0
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

```
The algorithm AdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3687 with standard deviation 0.0141.

took 31.77 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

