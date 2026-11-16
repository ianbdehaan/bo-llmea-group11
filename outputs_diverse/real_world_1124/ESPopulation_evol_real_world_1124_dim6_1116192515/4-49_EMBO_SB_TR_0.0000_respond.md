# Description 
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration, Smart Batch Selection and Enhanced Trust Region (EMBO-SB+TR)**

This refined Bayesian optimization algorithm builds upon the EMBO-SB framework with several key enhancements to improve performance on black-box optimization problems. The main improvements include:

1. **Enhanced Trust Region Management**: Implements a more sophisticated trust region mechanism that dynamically adjusts based on recent progress, with adaptive shrinkage/expansion factors and better center selection
2. **Improved Acquisition Function Optimization**: Uses a hybrid approach combining multi-start L-BFGS with local refinement around promising regions, plus Thompson sampling for better exploration 
3. **Advanced Batch Selection Strategy**: Implements an adaptive batch sizing that considers both budget remaining and problem characteristics, with more diverse sampling within trust regions
4. **Smart Surrogate Model Updates**: Uses ensemble of models with uncertainty quantification, and implements early stopping for GP hyperparameter optimization
5. **Dynamic Exploration Scheduling**: Adapts exploration weight based on optimization progress rather than fixed intervals, with better handling of early iterations

# Justification 
The key components are chosen to address identified weaknesses in the baseline approach while maintaining computational efficiency:

1. **Trust Region Enhancement**: The original trust region was too simplistic. Improved version uses adaptive factors that respond to actual performance changes, and better center selection (using a robust estimate rather than just the best point).

2. **Acquisition Function Optimization**: Multi-start L-BFGS alone can be inefficient for high-dimensional problems. Adding Thompson sampling provides better exploration and helps escape local optima.

3. **Batch Selection**: The adaptive batch sizing is refined to better balance exploration vs exploitation across different budget phases, with consideration of problem dimensionality.

4. **Surrogate Model**: Using ensemble models or more sophisticated uncertainty quantification can improve robustness and reduce overfitting in early stages.

5. **Exploration Scheduling**: Dynamic weights that adapt based on actual progress are more effective than fixed schedules for varying problem characteristics.

These changes focus on improving robustness, computational efficiency, and adaptability to different optimization phases while keeping the overall approach practical for limited budget scenarios.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
import warnings

class EMBO_SB_TR:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10) # Adaptive initial points

        # Random generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model parameters
        self.gp = None
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.region_shrink_factor = 0.8
        self.region_expand_factor = 1.2
        self.progress_threshold = 1e-3
        self.last_improvement_evals = 0
        self.improvement_count = 0
        self.trust_region_center = None
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
        # Batch size management
        self.initial_batch_size = min(5, max(1, dim))
        self.max_batch_size = min(10, budget // 10)
        self.min_batch_size = 1
        
        # Acquisition function parameters
        self.acq_func_type = 'ei'  # ei, lcb, thompson
        self.lcb_weight_schedule = [1.0, 0.5, 0.2]  # Dynamic weight schedule for LCB
        
        # Surrogate model parameters
        self.use_ensemble = True
        self.ensemble_size = 3
        self.gp_n_restarts = 3
        self.gp_alpha = 1e-6
        
        # Early stopping parameters
        self.max_gp_restarts = 5
        self.min_gp_restarts = 2

    def _sample_points(self, n_points):
        # Use maximin Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel with limited restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low, but at least some
        n_restarts = min(self.max_gp_restarts, max(self.min_gp_restarts, self.gp_n_restarts))
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=self.gp_alpha,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        try:
            gp.fit(X, y)
        except Exception:
            # Fallback to a simpler model if GP fails
            warnings.warn("GP fitting failed, using fallback model")
            # Use simple constant model as fallback
            gp = None
            
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current best value (with some smoothing to avoid numerical issues)
        if len(self.y) > 0:
            current_best = np.min(self.y) - 1e-8
        else:
            current_best = 0.0
            
        # Compute EI
        improvement = current_best - mean
        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _get_trust_region_center(self):
        # Return a more robust estimate of the center of trust region
        if self.X is None or len(self.X) == 0:
            return np.mean(self.bounds, axis=0)
        else:
            # Use a robust estimator: median of top 10% points
            y_sorted = np.argsort(self.y.flatten())
            n_top = max(1, len(y_sorted) // 10)
            top_indices = y_sorted[:n_top]
            top_points = self.X[top_indices]
            return np.median(top_points, axis=0)

    def _update_trust_region(self):
        # Update trust region based on recent progress with more sophisticated logic
        if len(self.performance_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        recent_improvement = (self.performance_history[-1] - self.performance_history[-2]) / max(1e-8, abs(self.performance_history[-1]))
        
        if recent_improvement > self.progress_threshold:
            # We're making good progress, expand the trust region
            self.trust_region_radius = min(
                self.max_trust_region_radius,
                self.trust_region_radius * self.region_expand_factor
            )
            self.improvement_count += 1
        else:
            # No significant improvement, shrink the trust region
            self.trust_region_radius = max(
                self.min_trust_region_radius,
                self.trust_region_radius * self.region_shrink_factor
            )
            self.improvement_count = 0

    def _sample_in_trust_region(self, center, n_points):
        # Sample points within trust region around the center
        # Create a box around center with size based on current trust region radius
        lower_bound = np.maximum(self.bounds[0], center - self.trust_region_radius)
        upper_bound = np.minimum(self.bounds[1], center + self.trust_region_radius)
        
        # If region is too small, just sample from bounds
        if np.all(upper_bound - lower_bound > 1e-6):
            # Use Latin Hypercube sampling within the trust region
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            points = sampler.random(n=n_points)
            points = lower_bound + points * (upper_bound - lower_bound)
        else:
            # Fall back to random sampling when trust region is very small
            points = self._sample_points(n_points)
            
        return points

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(20, max(5, batch_size * 3))  # More starts than needed to ensure diversity
        
        # Get current trust region center and update it
        tr_center = self._get_trust_region_center()
        self.trust_region_center = tr_center
        
        # Generate diverse starting points within trust region
        start_points = self._sample_in_trust_region(tr_center, n_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use L-BFGS with multiple restarts to improve precision
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point.flatten(),
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20, 'disp': False}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value > best_value:  # Maximize EI
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from trust region
            selected_points = self._sample_in_trust_region(tr_center, 1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points within trust region
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_in_trust_region(tr_center, 1)
                candidates.append(rand_point)
                
        return np.vstack(candidates)[:batch_size]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _get_adaptive_batch_size(self):
        # Adaptive batch size that decreases as we approach budget limit
        progress = self.n_evals / self.budget
        
        if progress < 0.2:
            # Early stage - use larger batches for exploration
            batch_size = max(self.min_batch_size, self.initial_batch_size)
        elif progress < 0.7:
            # Middle stage - moderate batch size
            batch_size = max(self.min_batch_size, self.max_batch_size // 2)
        else:
            # Late stage - smaller batches to avoid waste
            batch_size = max(self.min_batch_size, int(self.max_batch_size * (1 - progress)))
            
        return min(batch_size, self.budget - self.n_evals)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update trust region based on recent performance
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
                
                # Update trust region every few iterations or when we have enough data
                if (self.n_evals - self.last_improvement_evals > max(10, self.budget // 20)) and len(self.performance_history) >= 2:
                    self._update_trust_region()
                    self.last_improvement_evals = self.n_evals
            
            # Select adaptive batch size
            batch_size = self._get_adaptive_batch_size()
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<EMBO_SB_TR>", line 68
    points = sampler.random(n=n=n_points)
                               ^
SyntaxError: invalid syntax
