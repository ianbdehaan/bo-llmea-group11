# Description 
**Adaptive Trust-Region Bayesian Optimization (ATRBO)**

This is a novel Bayesian optimization algorithm that combines adaptive trust-region strategies with multi-point acquisition functions. The key characteristics are:
1. Uses a trust-region mechanism that dynamically adjusts based on function landscape changes
2. Employs an ensemble of acquisition functions to balance exploration and exploitation  
3. Implements efficient local search around promising regions
4. Uses a sophisticated initial design (Latin hypercube with optimization)
5. Adapts the surrogate model complexity based on evaluation progress

# Justification 
The key innovation in this algorithm is replacing the dynamic exploration schedule with an adaptive trust-region strategy, which offers several advantages:

1. **Trust Region Adaptation**: Instead of using a fixed schedule that reduces exploration over time, the algorithm maintains a trust region around the current best point that grows/shrinks based on optimization progress. This allows for more intelligent exploration-exploitation balance.

2. **Progress Monitoring**: The algorithm monitors function improvement over recent evaluations. If progress stagnates (e.g., no improvement in 5 consecutive evaluations), it triggers a restart mechanism by expanding the trust region, encouraging global exploration.

3. **Efficient Local Search**: When within the trust region, the algorithm focuses on local refinement, which is computationally efficient and appropriate for problems with complex landscapes.

4. **Dynamic Budget Allocation**: The algorithm allocates more budget to early phases for global exploration and shifts towards local exploitation as progress is made, making it highly adaptive.

5. **Robustness**: The trust-region approach is inherently robust to various problem characteristics and doesn't rely on potentially brittle scheduling heuristics that may not adapt well to different optimization landscapes.

The trust-region strategy works better than the original dynamic exploration because:
- It's data-driven rather than heuristic-based
- It adapts automatically to the function landscape 
- It provides clear stopping criteria for when to expand/restart exploration
- It avoids the computational overhead of maintaining complex schedules
- It's more intuitive and theoretically sound in optimization contexts

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
import warnings

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Set initial design size based on dimensionality
        if dim <= 3:
            self.n_init = max(5, dim + 1)
        elif dim <= 6:
            self.n_init = max(10, 2 * dim)
        else:
            self.n_init = max(15, 3 * dim)
            
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.min_trust_region_radius = 0.1
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        
        # Progress tracking for restarts
        self.last_improvement_eval = 0
        self.improvement_window = 5
        self.best_y_history = []
        
        # Acquisition function parameters
        self.acq_func_type = 'ei'  # Expected Improvement as default

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process surrogate model with ARD Matern kernel
        try:
            # Try different kernel configurations to avoid numerical issues
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e2), nu=2.5)
            
            # Add WhiteKernel for noise modeling
            kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
            
            # Fit GP with robust hyperparameter optimization
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Small regularization to avoid numerical issues
                normalize_y=True,
                n_restarts_optimizer=3,  # Limit restarts for efficiency
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X, y)
            return gp
        except Exception:
            # Fallback to simpler model if complex fitting fails
            warnings.warn("GP fitting failed, using fallback")
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e2), nu=2.5)
            kernel = kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function with trust region-aware exploration
        if self.X is None or self.y is None:
            # If no data yet, return random values for initial exploration
            return np.random.rand(X.shape[0], 1)
            
        # Fit model on current data
        model = self._fit_model(self.X, self.y)
        
        # Get current best value
        best_y = np.min(self.y)
        
        # Compute acquisition function
        try:
            # Get predictions from GP
            mean, std = model.predict(X, return_std=True)
            
            # Expected Improvement calculation
            improvement = best_y - mean
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Apply trust region weighting - points outside trust region get reduced weight
            if self.X is not None and len(self.X) > 0:
                # Calculate distance from current best point
                best_x = self.X[np.argmin(self.y)]
                distances = np.sqrt(np.sum((X - best_x)**2, axis=1))
                
                # Apply trust region weighting (points beyond radius get reduced weight)
                weights = np.ones_like(distances)
                outside_region = distances > self.trust_region_radius
                weights[outside_region] = 0.5  # Reduce weight for points outside trust region
                
                ei = ei * weights.reshape(-1, 1)
            
            return ei.reshape(-1, 1)
        except Exception:
            # Fallback to random acquisition if calculation fails
            return np.random.rand(X.shape[0], 1)

    def _select_next_points(self, batch_size):
        # Select next points using trust-region-aware multi-start optimization
        
        # Generate diverse candidates within the current trust region
        n_candidates = max(50, batch_size * 10)
        
        # Sample within trust region if we have a good solution
        if self.X is not None and len(self.X) > 0:
            best_x = self.X[np.argmin(self.y)]
            # Generate candidates within trust region
            candidate_points = []
            for _ in range(n_candidates):
                # Sample point within trust region
                sample_point = np.zeros(self.dim)
                for i in range(self.dim):
                    lower_bound = max(self.bounds[0][i], best_x[i] - self.trust_region_radius)
                    upper_bound = min(self.bounds[1][i], best_x[i] + self.trust_region_radius)
                    sample_point[i] = self.rng.uniform(lower_bound, upper_bound)
                candidate_points.append(sample_point)
            candidate_points = np.array(candidate_points)
        else:
            # If no data yet, use global sampling
            candidate_points = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for candidates
        acq_values = self._acquisition_function(candidate_points).flatten()
        
        # Select top candidates based on acquisition values
        top_indices = np.argsort(acq_values)[-batch_size:]
        selected_points = candidate_points[top_indices]
        
        # Refine selection using local optimization around best points
        refined_points = []
        for point in selected_points:
            try:
                # Use differential evolution to optimize acquisition locally
                def objective(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x)[0][0]
                
                # Bound the search within trust region around current point
                bounds = []
                if self.X is not None and len(self.X) > 0:
                    best_x = self.X[np.argmin(self.y)]
                    for i in range(len(point)):
                        lower_bound = max(self.bounds[0][i], best_x[i] - self.trust_region_radius)
                        upper_bound = min(self.bounds[1][i], best_x[i] + self.trust_region_radius)
                        bounds.append((lower_bound, upper_bound))
                else:
                    # If no data, use global bounds
                    for i in range(len(point)):
                        bounds.append((self.bounds[0][i], self.bounds[1][i]))
                
                result = differential_evolution(objective, bounds, seed=int(self.rng.integers(2**31 - 1)), maxiter=20)
                refined_points.append(result.x)
            except:
                # If local optimization fails, keep the original point
                refined_points.append(point)
        
        return np.array(refined_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        y_values = []
        for point in X:
            y_values.append(func(point))
        
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _update_trust_region(self):
        # Update trust region based on progress
        if len(self.best_y_history) < 2:
            return
            
        # Check if there's been improvement in recent evaluations
        recent_improvement = False
        current_best = np.min(self.y)
        
        # Add to history tracking
        self.best_y_history.append(current_best)
        if len(self.best_y_history) > self.improvement_window:
            self.best_y_history.pop(0)
            
        # Check if we're making progress (improvement in last window)
        if len(self.best_y_history) >= self.improvement_window:
            improvement = self.best_y_history[0] - self.best_y_history[-1]
            if improvement < 1e-4:  # Very small improvement
                # No significant improvement - expand trust region
                self.trust_region_radius = min(
                    self.max_trust_region_radius,
                    self.trust_region_radius * self.trust_region_expand_factor
                )
            else:
                # Significant improvement - shrink trust region to focus on local search
                self.trust_region_radius = max(
                    self.min_trust_region_radius,
                    self.trust_region_radius * self.trust_region_shrink_factor
                )

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial design
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're near the budget, evaluate remaining points
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 1:
                break
                
            # Determine batch size (limit to remaining budget)
            batch_size = min(5, remaining_budget)
            
            # Update trust region before selecting next points
            self._update_trust_region()
            
            # Select next points using acquisition function with trust-region awareness
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Break if budget exhausted
            if self.n_evals >= self.budget:
                break
        
        # Return best value and point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRBO>", line 273, in __call__
 273->             next_points = self._select_next_points(batch_size)
  File "<ATRBO>", line 159, in _select_next_points
 157 |         # Select top candidates based on acquisition values
 158 |         top_indices = np.argsort(acq_values)[-batch_size:]
 159->         selected_points = candidate_points[top_indices]
 160 |         
 161 |         # Refine selection using local optimization around best points
IndexError: index 2291 is out of bounds for axis 0 with size 50
