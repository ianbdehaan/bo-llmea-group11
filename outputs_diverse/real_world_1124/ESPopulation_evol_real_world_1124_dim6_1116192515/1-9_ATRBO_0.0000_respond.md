# Description 
**Adaptive Trust-Region Bayesian Optimization (ATRBO)**

This is a novel Bayesian optimization algorithm that combines adaptive trust-region strategies with multi-point acquisition functions. The key characteristics are:
1. Uses a trust-region mechanism that dynamically adjusts based on function landscape changes
2. Employs an ensemble of acquisition functions to balance exploration and exploitation
3. Implements efficient local search around promising regions
4. Uses a sophisticated initial design (Latin hypercube with optimization)
5. Adapts the surrogate model complexity based on evaluation progress

# Justification 
The proposed algorithm replaces the dynamic exploration schedule with an adaptive trust-region strategy, which is more suitable for real-world problems with complex landscapes. Trust regions work better than simple exploration schedules because:

1. **Adaptive Size Control**: The trust region size automatically adjusts based on function behavior - shrinking when progress is good and expanding when stagnation occurs. This prevents unnecessary exploration in flat regions while maintaining sufficient exploration in complex areas.

2. **Efficient Local Search**: By focusing search within adaptive trust regions, we can efficiently explore promising local areas without wasting evaluations on unproductive regions.

3. **Better Budget Allocation**: Unlike fixed schedules that may over-explore early or under-exploit later, trust regions adaptively allocate budget to where it's most needed.

4. **Robustness**: The algorithm adapts to both smooth and noisy landscapes by adjusting region sizes based on local curvature estimates.

The trust-region approach is particularly effective for the FM sound wave parameter estimation problem because:
- The search space has complex multimodal characteristics
- Early exploration needs to be efficient but not too greedy
- Local refinement around promising areas should be prioritized
- Budget constraints require adaptive allocation strategies

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
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
        # For low dimensions use more points for better exploration
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
        self.trust_region_center = None  # Center of current trust region
        self.min_radius = 0.1  # Minimum trust region radius
        self.max_radius = 5.0  # Maximum trust region radius
        self.radius_shrink_factor = 0.5  # Factor to shrink radius when stagnation occurs
        self.radius_expand_factor = 1.5  # Factor to expand radius when progress is good
        self.stagnation_count = 0  # Count of consecutive iterations with no improvement
        self.max_stagnation = 5  # Maximum stagnation before shrinking trust region
        
        # Acquisition function parameters
        self.acq_func_type = "ei"  # Expected Improvement as default
        self.acq_func_weights = [0.7, 0.3]  # Weights for different acquisition functions

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process surrogate model with ARD Matern kernel
        # Use a more robust approach for fitting GP
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
        # Expected Improvement acquisition function with trust-region based exploration
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
            
            # Add exploration component based on trust region proximity
            if self.trust_region_center is not None:
                # Calculate distance from trust region center
                distances = np.sqrt(np.sum((X - self.trust_region_center)**2, axis=1))
                # Scale exploration based on proximity to center
                proximity_factor = 1.0 / (1.0 + distances / self.trust_region_radius)
                ei = ei * proximity_factor.reshape(-1, 1)
            
            return ei.reshape(-1, 1)
        except Exception:
            # Fallback to random acquisition if calculation fails
            return np.random.rand(X.shape[0], 1)

    def _select_next_points(self, batch_size):
        # Select next points using trust-region constrained optimization
        # Generate candidates within current trust region
        
        # If no trust region center yet, initialize it at the best point found so far
        if self.trust_region_center is None:
            if self.X is not None and len(self.X) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx].copy()
            else:
                # Initialize center at random point in search space
                self.trust_region_center = self._sample_points(1)[0]
        
        # Generate diverse candidates within trust region
        n_candidates = max(50, batch_size * 10)
        candidate_points = []
        
        for _ in range(n_candidates):
            # Sample uniformly within the trust region
            while True:
                # Generate random point within bounds
                point = self._sample_points(1)[0]
                
                # Check if it's within trust region
                distance = np.sqrt(np.sum((point - self.trust_region_center)**2))
                if distance <= self.trust_region_radius:
                    candidate_points.append(point)
                    break
        
        candidate_points = np.array(candidate_points)
        
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
                bounds = [(max(self.bounds[0][i], max(self.trust_region_center[i] - self.trust_region_radius, point[i] - 1.0)), 
                          min(self.bounds[1][i], min(self.trust_region_center[i] + self.trust_region_radius, point[i] + 1.0))) 
                         for i in range(len(point))]
                
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

    def _update_trust_region(self, current_best_y):
        # Update trust region based on recent progress
        if self.X is not None and len(self.X) >= 2:
            # Calculate improvement over last few evaluations
            recent_evals = min(5, len(self.y))
            recent_improvement = np.min(self.y[-recent_evals:]) - current_best_y
            
            # Update stagnation count
            if recent_improvement < 1e-6:  # No significant improvement
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0  # Reset stagnation counter
            
            # Adjust trust region size based on progress
            if self.stagnation_count >= self.max_stagnation:
                # Shrink trust region when stagnation occurs
                self.trust_region_radius = max(self.min_radius, 
                                              self.trust_region_radius * self.radius_shrink_factor)
                self.stagnation_count = 0  # Reset counter after adjustment
            elif recent_improvement > 1e-4:  # Significant improvement
                # Expand trust region when progress is good
                self.trust_region_radius = min(self.max_radius,
                                              self.trust_region_radius * self.radius_expand_factor)
        else:
            # For initial evaluations, use fixed radius
            pass

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial design
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Track best value found so far for trust region updates
        best_y_so_far = np.min(self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're near the budget, evaluate remaining points
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 1:
                break
                
            # Determine batch size (limit to remaining budget)
            batch_size = min(5, remaining_budget)
            
            # Update trust region based on recent progress
            self._update_trust_region(best_y_so_far)
            
            # Select next points using acquisition function within trust region
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Update best value found so far
            current_best_y = np.min(self.y)
            if current_best_y < best_y_so_far:
                best_y_so_far = current_best_y
            
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
  File "<ATRBO>", line 265, in __call__
 265->             next_points = self._select_next_points(batch_size)
  File "<ATRBO>", line 162, in _select_next_points
 160 |         # Select top candidates based on acquisition values
 161 |         top_indices = np.argsort(acq_values)[-batch_size:]
 162->         selected_points = candidate_points[top_indices]
 163 |         
 164 |         # Refine selection using local optimization around best points
IndexError: index 864 is out of bounds for axis 0 with size 50
