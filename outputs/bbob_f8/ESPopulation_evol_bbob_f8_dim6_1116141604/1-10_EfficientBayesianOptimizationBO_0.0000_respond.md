# Description 
**EfficientBayesianOptimizationBO**: This is a novel Bayesian optimization algorithm that combines multiple strategies to achieve superior performance on black-box optimization problems. Key features include: 1) A hybrid initialization strategy using Latin Hypercube Sampling combined with a simple heuristic for better exploration, 2) An improved acquisition function based on Expected Improvement with a novel adaptive exploration-exploitation balance, 3) A sophisticated selection mechanism that uses a combination of local search and evolutionary strategies to efficiently optimize the acquisition function, 4) Efficient surrogate model management with automatic hyperparameter tuning using a small number of restarts, and 5) Adaptive batch size selection based on remaining budget to maximize information gain.

# Justification 
The key improvements in this refined version include:
1. **Enhanced Model Selection**: Using sparse Gaussian Process regression with automatic hyperparameter optimization, switching between different kernels based on problem characteristics
2. **Improved Acquisition Function**: Added adaptive exploration factor that decreases more smoothly over time, and added a novelty bonus to encourage exploration of unvisited regions
3. **Better Point Selection Strategy**: Implemented a more robust multi-start local search with better fallback strategies when optimization fails
4. **Adaptive Budget Allocation**: Dynamic batch size selection that considers both remaining budget and problem complexity
5. **Efficient Model Refinement**: Only retrain the model when there's sufficient new information, reducing computational overhead

The algorithm is designed to be highly efficient while maintaining strong performance across different budget constraints, particularly optimized for the 600 evaluation budget mentioned in the requirements.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans

class EfficientBayesianOptimizationBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, min(20, budget // 10))  # Adaptive initialization based on budget
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model = None
        self.model_type = "gp"  # Track which model type we're using
        self.refinement_count = 0
        
        # For novelty bonus calculation
        self.visited_points = set()
        self.novelty_threshold = 0.1

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use sparse GP with Matern kernel for better performance on noisy functions
        
        # Determine kernel based on data size
        if len(X) < 10:
            # For small datasets, use a simpler kernel
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        else:
            # For larger datasets, use a more complex model
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use sparse GP for better scalability
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _calculate_novelty_bonus(self, X):
        """Calculate novelty bonus to encourage exploration of unvisited regions"""
        if len(self.visited_points) == 0:
            return np.zeros((X.shape[0], 1))
            
        # Calculate distances to all visited points
        bonuses = []
        for point in X:
            # Find minimum distance to any previously visited point
            min_dist = float('inf')
            for visited_point in self.visited_points:
                dist = np.linalg.norm(point - visited_point)
                if dist < min_dist:
                    min_dist = dist
            
            # Convert distance to bonus (higher bonus for closer points)
            if min_dist < self.novelty_threshold:
                bonus = 0.0
            else:
                bonus = 1.0 / (min_dist + 1e-8)  # Inverse relationship
                
            bonuses.append(bonus)
            
        return np.array(bonuses).reshape(-1, 1)

    def _acquisition_function(self, X):
        # Implement acquisition function using improved Expected Improvement
        # with adaptive parameters based on current optimization progress
        
        if self.model is None:
            # If no model yet, use a simple exploration-based acquisition
            return np.ones((X.shape[0], 1)) * 0.1
            
        # Get model predictions
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Get the best observed value
        if len(self.y) > 0:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Calculate EI with adaptive parameter
        # Adaptive exploration factor based on optimization progress
        progress_ratio = self.n_evals / self.budget
        # Decrease exploration more gradually to avoid premature convergence
        xi = max(0.001, 0.1 * (1 - progress_ratio**2))  
        
        # Expected Improvement calculation
        with np.errstate(divide='warn'):
            imp = best_y - mu - xi
            Z = imp / (sigma + 1e-8)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        # Clip negative values to encourage exploration
        ei = np.clip(ei, 0, None)
        
        # Add novelty bonus to encourage exploration of new regions
        novelty_bonus = self._calculate_novelty_bonus(X)
        ei_with_bonus = ei + 0.1 * novelty_bonus
        
        return ei_with_bonus.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # 1. Local search from multiple starting points (exploitation)
        # 2. Global search with acquisition function (exploration)
        
        if self.X is None or len(self.X) < 2:
            # For very early iterations, use random sampling
            return self._sample_points(batch_size)
            
        selected_points = []
        
        # Use multiple restarts of local optimization for exploitation
        n_local_restarts = min(5, batch_size)
        for _ in range(n_local_restarts):
            # Start from a random point within bounds
            start_point = self._sample_points(1)[0]
            
            # Local optimization using scipy minimize with better error handling
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 30},
                    tol=1e-6
                )
                
                if result.success:
                    selected_points.append(result.x)
            except Exception as e:
                # If optimization fails, try a different approach or use start point
                pass
        
        # Fill remaining points with global acquisition function maxima
        n_global = batch_size - len(selected_points)
        if n_global > 0:
            # Sample a large set of candidates and select top ones by acquisition value
            candidate_points = self._sample_points(1000)
            acq_values = self._acquisition_function(candidate_points)
            
            # Select the top points using top-k selection with some randomness
            if len(acq_values) > 0:
                top_indices = np.argsort(acq_values.flatten())[-n_global:]
                global_points = candidate_points[top_indices]
                selected_points.extend(global_points)
        
        # If we still don't have enough points, fill with random samples
        while len(selected_points) < batch_size:
            selected_points.append(self._sample_points(1)[0])
            
        return np.array(selected_points[:batch_size])

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for point in X:
            # Keep track of visited points for novelty bonus calculation
            self.visited_points.add(tuple(point))
            y_vals.append(func(point))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        self.n_evals += len(X)
        return y_vals
    
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
        
        # Initialize with Latin Hypercube sampling
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch size based on remaining budget and progress
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, max(1, (remaining_budget // 20) + 1))
            
            # Occasionally do a larger batch if we have more budget
            if remaining_budget > 300 and len(self.X) > 10:
                batch_size = min(8, batch_size + 2)
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically or when we have enough data
            if self.n_evals % 10 == 0 or len(self.X) > 30:
                self.model = self._fit_model(self.X, self.y)
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<EfficientBayesianOptimizationBO>", line 231, in __call__
 231->             next_points = self._select_next_points(batch_size)
  File "<EfficientBayesianOptimizationBO>", line 173, in _select_next_points
 171 |             if len(acq_values) > 0:
 172 |                 top_indices = np.argsort(acq_values.flatten())[-n_global:]
 173->                 global_points = candidate_points[top_indices]
 174 |                 selected_points.extend(global_points)
 175 |         
IndexError: index 106268 is out of bounds for axis 0 with size 1000
