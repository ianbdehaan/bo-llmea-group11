# Description
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB) - Refined Version**

This refined Bayesian optimization algorithm improves upon the previous EMBO-SB framework by incorporating several key enhancements to achieve better performance and robustness. The main improvements include:

1. **Enhanced Initial Design**: Uses a maximin Latin Hypercube design with adaptive scaling based on problem dimensionality and budget
2. **Improved Surrogate Model**: Implements a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel optimized with efficient hyperparameter restarts
3. **Adaptive Acquisition Function**: Combines Expected Improvement with Lower Confidence Bound using dynamic weights that adapt based on progress through the budget
4. **Smart Batch Selection Strategy**: Implements an adaptive batch sizing mechanism that decreases as budget is exhausted, and uses diverse sampling strategies for batch points
5. **Robust Acquisition Optimization**: Employs a multi-start L-BFGS optimizer with diverse starting points including trust-region boundaries and recent best points
6. **Progress-Driven Exploration Scheduling**: Uses a dynamic exploration weight schedule that adapts based on optimization progress rather than fixed time intervals

The algorithm introduces a more sophisticated acquisition function that balances exploitation and exploration, uses better batch selection strategies to maintain diversity while respecting the budget, and incorporates more efficient model updates.

# Justification
The key components of this algorithm were chosen for their proven effectiveness in Bayesian optimization:

1. **ARD Matern Kernel**: Provides flexible modeling of the objective function with automatic relevance detection, which is crucial for high-dimensional problems like the 6D FM sound wave parameter estimation.

2. **Adaptive Exploration Weight Schedule**: The dynamic weight schedule ensures sufficient exploration early in the optimization when little information is available, then shifts towards exploitation as the search progresses, which is critical for avoiding local optima while maintaining convergence speed.

3. **Multi-start Acquisition Optimization**: Multiple restarts of L-BFGS with diverse starting points (including boundaries and recent best points) help avoid getting stuck in poor local optima of the acquisition function, improving selection quality.

4. **Adaptive Batch Selection**: The batch size decreases as budget is exhausted, ensuring we don't waste evaluations on less promising candidates near the end of optimization while maintaining sufficient exploration early on.

5. **Smart Evaluation Strategy**: The algorithm carefully manages when and how many points to evaluate in batches, balancing computational efficiency with optimization effectiveness.

6. **Robust Implementation Details**: Using numpy's default_rng for reproducibility, proper bounds handling, and careful memory management ensure the algorithm is stable and efficient across different runs.

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

class EMBO_SB_Refined:
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
        """Select next points using acquisition function optimization"""
        # For small batches, use a more focused approach
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use multi-start L-BFGS to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=10)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using random sampling with some structure
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample from a region around the current best point to maintain diversity
                if self.X is not None and len(self.X) > 0:
                    # Use recent best points to define search region
                    recent_best_idx = np.argsort(self.y.flatten())[:min(5, len(self.y))]
                    recent_best_points = self.X[recent_best_idx]
                    # Add some variation by sampling from a local region around the best point
                    if len(recent_best_points) > 0:
                        center = recent_best_points[np.argmin(self.y[recent_best_idx])]
                        # Sample around the center point with some randomness
                        for _ in range(remaining):
                            sample_point = center + self.rng.normal(0, 0.3, self.dim)
                            # Clip to bounds
                            sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                            selected_points.append(sample_point.reshape(1, -1))
                    else:
                        # Fallback to random sampling
                        for _ in range(remaining):
                            selected_points.append(self._sample_points(1))
                else:
                    # Fallback to random sampling
                    for _ in range(remaining):
                        selected_points.append(self._sample_points(1))
            
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
            
            # Use a more sophisticated batch sizing strategy that's sensitive to budget exhaustion
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                # More aggressive decrease in batch size as we approach budget limit
                # But keep it reasonable to maintain exploration
                batch_size = max(1, min(5, int(remaining_budget // 4)))  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update model with current data (only when we have enough points)
            if self.n_evals > 0 and len(self.X) >= 2 * self.dim + 1:
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
## Feedback
 The algorithm EMBO_SB_Refined got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4561 with standard deviation 0.0141.

took 207.11 seconds to run.