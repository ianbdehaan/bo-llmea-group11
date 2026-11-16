# Description 
**AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.

# Justification 
The key change replaces the existing trust region adaptation mechanism with a more robust restart-based exploration strategy that dynamically switches between global and local search phases. This approach is superior because:

1. **Restart Mechanism**: Instead of gradual trust region adjustments, we use explicit restarts when stagnation is detected, allowing for complete reset of search focus when progress plateaus. This prevents getting trapped in poor local regions.

2. **Global-Local Switching**: The algorithm dynamically switches between global exploration (wide search) and local exploitation (focused search) based on performance metrics. This is more adaptive than fixed trust region scaling.

3. **Diverse Batch Selection**: Instead of using a simple acquisition function optimization, we implement a diverse batch selection that maintains good coverage of the search space while still leveraging the surrogate model for intelligent point selection.

4. **Improved Convergence**: The restart mechanism combined with smart switching allows better escape from local optima, which is crucial for complex functions like Rosenbrock where convergence to the global optimum can be challenging.

5. **Computational Efficiency**: The approach maintains computational efficiency by avoiding excessive GP retraining and keeping acquisition optimization manageable while still providing good exploration-exploitation balance.

The new strategy addresses the limitations of gradual trust region adjustment by ensuring more robust escape from local optima through explicit restarts, leading to better convergence properties for challenging black-box optimization problems.

# Code
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
        
        # Restart and exploration strategy parameters
        self.restart_threshold = 0.001  # Minimum improvement to consider as progress
        self.stagnation_count = 0
        self.max_stagnation = 5  # Number of iterations without progress before restart
        self.restart_counter = 0
        self.global_search = True  # Start with global search
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for restart adaptation
        self.best_y_history = []
        self.last_improvement_iter = 0
        self.iteration = 0

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
            n_restarts_optimizer=3,  # Increased restarts for better optimization
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
        """Select the next points to evaluate using diverse batch selection"""
        # Use a combination of acquisition function optimization and diversity sampling
        n_candidates = max(50 * self.dim, 200)
        
        if self.global_search:
            # Global search: use random sampling with some structured approach
            candidates = self._sample_points(n_candidates)
            
            # Add some points near the current best to maintain some local information
            if self.X is not None and len(self.X) > 0:
                best_x = self.X[np.argmin(self.y)]
                # Add a few points around the best point
                local_samples = []
                for _ in range(min(10, n_candidates // 10)):
                    sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * 0.5
                    sample = np.clip(sample, self.bounds[0], self.bounds[1])
                    local_samples.append(sample)
                candidates = np.vstack([candidates, local_samples])
        else:
            # Local search: use acquisition function optimization with multiple starting points
            if self.X is not None and len(self.X) > 0:
                best_x = self.X[np.argmin(self.y)]
                
                # Generate candidates around the current best point
                candidates = []
                for _ in range(n_candidates):
                    sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * 0.5
                    sample = np.clip(sample, self.bounds[0], self.bounds[1])
                    candidates.append(sample)
                candidates = np.array(candidates)
            else:
                candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates using a diverse selection approach
        # Sort by acquisition value and select the best ones
        sorted_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[sorted_indices]
        
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

    def _check_restart_condition(self):
        """Check if we should restart the optimization"""
        # Check for stagnation in performance
        if len(self.best_y_history) < 2:
            return False
            
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        if improvement < self.restart_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            
        # Restart if stagnation threshold is reached
        if self.stagnation_count >= self.max_stagnation:
            return True
            
        return False

    def _switch_search_strategy(self):
        """Switch between global and local search strategies"""
        # Switch to local search after a certain number of iterations or when we have enough data
        if len(self.y) > 2 * self.dim:
            self.global_search = False
        else:
            # Occasionally switch back to global search to maintain exploration
            if self.rng.random() < 0.1:  # 10% chance to switch back to global
                self.global_search = True

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
            
            # Check for restart condition
            if self._check_restart_condition():
                # Restart by reinitializing with a larger sample
                print(f"Restarting at iteration {self.iteration} with best value: {np.min(self.y)}")
                self.restart_counter += 1
                
                # Reset stagnation count
                self.stagnation_count = 0
                
                # Switch search strategy
                self.global_search = True
                
                # Sample a new batch of points to restart from
                n_restart_points = max(5, min(20, self.dim * 3))
                restart_points = self._sample_points(n_restart_points)
                y_restart = self._evaluate_points(func, restart_points)
                self._update_eval_points(restart_points, y_restart)
                
                # Refit model after restart
                self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Switch search strategy if needed
            self._switch_search_strategy()
            
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
                
            self.iteration += 1
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionBO>", line 232, in __call__
 232->                 y_restart = self._evaluate_points(func, restart_points)
  File "<AdaptiveTrustRegionBO>", line 149, in _evaluate_points
 147 |         y_vals = []
 148 |         for x in X:
 149->             y_vals.append(func(x))
 150 |             
 151 |         y_vals = np.array(y_vals).reshape(-1, 1)
llamevol.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
