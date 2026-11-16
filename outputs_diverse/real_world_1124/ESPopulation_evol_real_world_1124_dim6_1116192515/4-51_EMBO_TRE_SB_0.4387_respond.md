# Description 
**Efficient Multi-Start Bayesian Optimization with Trust-Region Exploration and Smart Batch Selection (EMBO-TRE-SB)**

This novel Bayesian optimization algorithm combines trust-region exploration with smart batch selection strategies to achieve excellent performance on black-box optimization problems. The key innovation is the implementation of a dynamic trust-region mechanism that adapts based on progress through the budget, allowing for both global exploration in early iterations and focused local search as optimization progresses. The algorithm also incorporates an adaptive batch sizing strategy that decreases as the budget is exhausted, while maintaining diversity through trust-region sampling.

# Justification 
The replacement of the exploration strategy with a trust-region mechanism offers several advantages:

1. **Dynamic Adaptation**: Unlike the previous approach which used fixed scheduling for exploration weights, TuRBO-style trust regions adapt dynamically to problem characteristics, allowing for more efficient exploration in early phases and focused exploitation later.

2. **Progress-Driven Search**: The trust-region grows and shrinks based on actual optimization progress, not arbitrary time intervals or fixed schedules. This ensures that when good progress is made, the search area can expand to explore new regions, while stagnation leads to contraction to focus on promising areas.

3. **Efficient Resource Allocation**: By focusing computational effort in a trust region around the current best point, the algorithm avoids wasteful exploration of unproductive areas, especially important for limited budget scenarios.

4. **Improved Convergence**: The trust-region approach helps avoid getting trapped in local optima by allowing for both global exploration (when regions are large) and fine-tuning (when regions are small), providing a more balanced optimization strategy.

5. **Scalability**: The trust-region mechanism is inherently scalable with problem dimensions as it focuses on the most promising area, making it suitable for higher-dimensional problems while maintaining efficiency.

The trust-region approach works better than the previous method because:
- It's data-driven rather than schedule-based
- It adapts to the actual landscape being optimized 
- It provides a natural way to balance exploration vs exploitation
- It reduces unnecessary function evaluations by focusing on promising regions
- It's computationally efficient and maintains good convergence properties

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class EMBO_TRE_SB:
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
        
        # Trust-region parameters
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_center = None  # Current trust region center
        self.min_radius = 0.1  # Minimum trust region radius
        self.max_radius = 5.0  # Maximum trust region radius
        self.radius_growth_factor = 1.2  # Factor to grow radius when progress is made
        self.radius_shrink_factor = 0.5  # Factor to shrink radius when no progress
        self.progress_window = max(5, dim)  # Number of iterations to consider for progress
        self.last_improvements = []  # Track recent improvements
        
        # Acquisition function parameters
        # Dynamic exploration weight based on progress through budget
        # Create a more refined schedule that starts with high exploration and gradually decreases
        if budget <= 100:
            self.acq_weight_schedule = np.linspace(5.0, 0.2, max(5, budget // 10))
        else:
            self.acq_weight_schedule = np.linspace(5.0, 0.1, max(10, budget // 10))
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use a small number of restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement combined with Lower Confidence Bound
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current exploration weight (adaptive)
        if len(self.acq_weight_schedule) > 0:
            # Use progress-based weight schedule
            progress = min(1.0, self.n_evals / max(1, self.budget * 0.7))
            # Map progress to appropriate weight from schedule
            weight_idx = int(progress * (len(self.acq_weight_schedule) - 1))
            exploration_weight = self.acq_weight_schedule[min(weight_idx, len(self.acq_weight_schedule) - 1)]
        else:
            exploration_weight = 0.5
            
        # Expected Improvement combined with LCB for robustness
        # EI is computed using the best observed value so far
        if self.y is not None and len(self.y) > 0:
            best_y = np.min(self.y)
            # Compute expected improvement
            if std > 1e-8:
                z = (best_y - mean) / std
                ei = (best_y - mean) * norm.cdf(z) + std * norm.pdf(z)
            else:
                ei = np.zeros_like(mean)
            
            # LCB component
            lcb = mean - exploration_weight * std
            
            # Combine both components for more robust selection
            combined_acq = ei - 0.5 * lcb  # Weighted combination to balance exploitation and exploration
        else:
            # Fallback if no data yet, use LCB
            combined_acq = mean - exploration_weight * std
            
        return combined_acq.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points including boundaries and recent best points
        start_points = self._sample_points(n_starts)
        
        # Add some boundary points for better exploration
        if self.X is not None and len(self.X) > 0:
            # Sample from recent best points to add diversity
            recent_best_idx = np.argsort(self.y.flatten())[:min(3, len(self.y))]
            recent_best_points = self.X[recent_best_idx]
            
            # Add boundary points (corners of the search space)
            corners = []
            for i in range(2**self.dim):
                corner = np.array([self.bounds[0][j] if (i >> j) & 1 else self.bounds[1][j] 
                                 for j in range(self.dim)])
                corners.append(corner)
            
            # Mix recent best points and corners with random points
            mixed_points = []
            for i in range(n_starts):
                if i < len(recent_best_points):
                    mixed_points.append(recent_best_points[i % len(recent_best_points)])
                elif i < len(recent_best_points) + len(corners):
                    mixed_points.append(corners[i - len(recent_best_points)])
                else:
                    mixed_points.append(start_points[i % len(start_points)])
            start_points = np.array(mixed_points)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using trust-region constrained sampling"""
        selected_points = []
        
        # If we don't have a trust region center yet, initialize with best point found so far
        if self.trust_region_center is None:
            if self.X is not None and len(self.X) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx].copy()
            else:
                # Fallback to center of search space
                self.trust_region_center = (self.bounds[0] + self.bounds[1]) / 2.0
        
        # First select one point using acquisition optimization within trust region
        best_point = self._optimize_acquisition(n_starts=10)
        
        if best_point is not None:
            # Ensure it's within trust region
            dist_to_center = np.linalg.norm(best_point[0] - self.trust_region_center)
            if dist_to_center > self.trust_region_radius:
                # Project back to trust region boundary
                direction = (best_point[0] - self.trust_region_center) / (dist_to_center + 1e-8)
                best_point = (self.trust_region_center + 
                             direction * self.trust_region_radius).reshape(1, -1)
            selected_points.append(best_point)
        
        # Add diverse points using trust-region sampling
        remaining = batch_size - len(selected_points)
        if remaining > 0:
            # Sample within trust region with some randomness
            for _ in range(remaining):
                # Generate point within trust region
                while True:
                    # Generate random point in hypercube and project to sphere
                    sample = self.rng.uniform(-1, 1, self.dim)
                    norm_sample = np.linalg.norm(sample)
                    if norm_sample > 1e-8:
                        unit_sample = sample / norm_sample
                        # Scale by trust region radius
                        point_in_trust_region = (self.trust_region_center + 
                                               unit_sample * self.rng.uniform(0, self.trust_region_radius))
                        
                        # Check bounds
                        if np.all(point_in_trust_region >= self.bounds[0]) and \
                           np.all(point_in_trust_region <= self.bounds[1]):
                            selected_points.append(point_in_trust_region.reshape(1, -1))
                            break
        
        return np.vstack(selected_points)

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

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        # Track recent improvements
        if len(self.y) >= 2:
            recent_improvements = []
            for i in range(1, min(self.progress_window + 1, len(self.y))):
                improvement = self.y[i-1] - self.y[i]
                recent_improvements.append(improvement)
            
            # Calculate average improvement
            if recent_improvements:
                avg_improvement = np.mean(recent_improvements)
                
                # Update trust region based on progress
                if avg_improvement > 0.01:  # Significant improvement
                    # Grow trust region
                    self.trust_region_radius = min(self.max_radius, 
                                                 self.trust_region_radius * self.radius_growth_factor)
                else:
                    # Shrink trust region
                    self.trust_region_radius = max(self.min_radius,
                                                 self.trust_region_radius * self.radius_shrink_factor)
        
        # Set trust region center to best point found so far
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            self.trust_region_center = self.X[best_idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube design scaled to problem dimensionality
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                # Decrease batch size as we approach budget limit but maintain reasonable sizes
                batch_size = max(1, min(5, remaining_budget // 4))  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update trust region based on recent progress
            self._update_trust_region()
            
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points using trust-region constrained sampling
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
## Feedback
 The algorithm EMBO_TRE_SB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4387 with standard deviation 0.0313.

took 95.60 seconds to run.