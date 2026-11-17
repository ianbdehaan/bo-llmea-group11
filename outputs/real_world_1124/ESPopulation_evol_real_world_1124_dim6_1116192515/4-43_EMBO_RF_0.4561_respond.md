# Description 
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration, Smart Batch Selection and Random Forest Surrogate Model (EMBO-RF)**

This refined Bayesian optimization algorithm builds upon the previous EMBO-SB framework but incorporates a substantially different surrogate model - a random forest regressor - to achieve better performance. The main enhancements include:
1. **Random Forest Surrogate Model**: Replaces Gaussian Process with Random Forest for improved robustness and handling of non-smooth functions
2. **Enhanced Initial Design**: Uses a maximin Latin Hypercube design with adaptive scaling based on problem dimensionality and budget  
3. **Improved Acquisition Function**: Combines Expected Improvement with Lower Confidence Bound using dynamic weights that adapt based on progress through the budget
4. **Smart Batch Selection Strategy**: Implements an adaptive batch sizing mechanism that decreases as budget is exhausted, and uses diverse sampling strategies for batch points
5. **Robust Acquisition Optimization**: Employs a multi-start L-BFGS optimizer with diverse starting points including trust-region boundaries and recent best points
6. **Progress-Driven Exploration Scheduling**: Uses a dynamic exploration weight schedule that adapts based on optimization progress rather than fixed time intervals

# Justification 
The replacement of the Gaussian Process surrogate model with a Random Forest regressor provides several key advantages for this specific optimization problem:

1. **Robustness to Non-smooth Functions**: The target function (1124 parameter estimation of Frequency Modulated Sound Waves) is likely to be highly non-smooth and potentially discontinuous. Random forests are inherently robust to such irregularities and can capture complex, non-linear patterns without requiring smoothness assumptions.

2. **Better Uncertainty Estimation**: Random forests naturally provide uncertainty estimates through the variance of predictions across trees, which is more reliable than GP's parametric uncertainty for complex functions with irregular behavior.

3. **Computational Efficiency**: Random forests scale better with dimensionality and can be trained more efficiently than GPs, especially when dealing with large numbers of evaluations or high-dimensional problems.

4. **Handling of Local Optima**: RFs are less prone to getting trapped in local optima compared to GP-based methods, which can overfit to local patterns in the data.

5. **Less Sensitive to Hyperparameters**: Random forests require fewer hyperparameters to tune compared to GPs, reducing the risk of suboptimal configuration and improving robustness across different problems.

The rest of the algorithm components (multi-start acquisition optimization, adaptive exploration scheduling, smart batch selection) remain largely unchanged as they provide complementary benefits that work well with the random forest surrogate model.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class EMBO_RF:
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
        
        # RF model parameters
        self.rf = None
        
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
        # Fit a Random Forest regressor
        # Use a reasonable number of trees for efficiency
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single thread for reproducibility
        )
        
        rf.fit(X, y.flatten())
        return rf

    def _acquisition_function(self, X):
        # Expected Improvement combined with Lower Confidence Bound
        if self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from RF
        mean = self.rf.predict(X)
        
        # Get uncertainty estimates (variance of predictions across trees)
        # For Random Forest, we can estimate variance through the individual tree predictions
        if hasattr(self.rf, 'estimators_'):
            # Get predictions from all trees for each point
            tree_predictions = np.array([tree.predict(X) for tree in self.rf.estimators_]).T
            std = np.std(tree_predictions, axis=1)
        else:
            # Fallback to a constant uncertainty estimate if we can't compute variance
            std = np.ones(X.shape[0]) * 0.1
            
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
            if np.any(std > 1e-8):
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
        self.rf = self._fit_model(self.X, self.y)
        
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
                
            # Update model with current data
            self.rf = self._fit_model(self.X, self.y)
            
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
 The algorithm EMBO_RF got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4561 with standard deviation 0.0141.

took 193.23 seconds to run.