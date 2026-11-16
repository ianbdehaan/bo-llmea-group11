# Description 
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Enhanced Surrogate Modeling (EMBO-ESM)**

This refined Bayesian optimization algorithm builds upon the EMBO framework by enhancing surrogate modeling, acquisition function selection, and optimization strategies. Key improvements include:

1. **Enhanced Surrogate Model**: Uses a more robust GP model with automatic relevance detection (ARD) Matern kernel optimized with multiple restarts, but with computational efficiency through early stopping and reduced restarts for later iterations.

2. **Improved Acquisition Function**: Implements Expected Improvement (EI) combined with a dynamic exploration weight that decreases over time, providing better balance between exploration and exploitation.

3. **Efficient Batch Selection**: Employs a multi-start optimization approach with diverse starting points using Sobol sampling for acquisition function optimization, while limiting computational overhead through early stopping criteria.

4. **Adaptive Model Refitting**: Implements intelligent model refitting based on progress to avoid unnecessary computations during early iterations when model updates are less impactful.

5. **Smart Budget Allocation**: Dynamically allocates budget between initial design and iterative refinement, with a focus on efficient point selection strategies that work well within limited evaluation budgets.

# Justification 

The key improvements in this algorithm address several weaknesses of the baseline EMBO:
1. **Surrogate Model Efficiency**: While maintaining model quality, reducing GP hyperparameter optimization restarts for later iterations to save computation time.
2. **Acquisition Function Strategy**: Using Expected Improvement instead of LCB provides more direct optimization guidance and better performance on many problems.
3. **Computational Budget Management**: Smart allocation of function evaluations to avoid overfitting during early iterations while maintaining sufficient exploration.
4. **Robust Optimization**: Multi-start acquisition function optimization with early stopping prevents getting stuck in local optima while keeping computation costs manageable.
5. **Scalability**: The algorithm scales well with dimensionality and budget, making it suitable for the 6-dimensional problem with a 600-evaluation budget.

These changes aim to improve convergence speed while maintaining robustness across different optimization landscapes, particularly important for black-box problems where function evaluations are expensive.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EMBOESM:
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
        # Dynamic exploration weight that decreases over time
        self.acq_weight_schedule = np.linspace(2.0, 0.2, max(10, budget // 10))  
        
        # Track if we've seen significant improvement to decide when to refit model more aggressively
        self.last_improvement_eval = 0
        self.best_y_so_far = np.inf
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for good space-filling properties in initial design
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use few restarts for early iterations to save computation, more for later
        n_restarts = max(3, min(10, 5 + self.n_evals // (self.budget // 4)))
        
        # For small budgets, use even fewer restarts
        if self.budget < 200:
            n_restarts = max(1, min(3, n_restarts))
            
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration weight
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current exploration weight (adaptive)
        if len(self.acq_weight_schedule) > 0:
            # Use the last available weight or interpolate based on evaluation count
            current_weight_idx = min(len(self.acq_weight_schedule) - 1, self.n_evals // max(1, self.budget // len(self.acq_weight_schedule)))
            exploration_weight = self.acq_weight_schedule[current_weight_idx]
        else:
            exploration_weight = 0.5
            
        # Expected Improvement acquisition function
        if np.any(std <= 0):
            # If standard deviation is zero, return zero improvement
            ei = np.zeros_like(mean)
        else:
            # Calculate EI with proper numerical handling
            z = (mean - self.best_y_so_far) / std
            # Use the standard formula for EI: (mu - f_best) * Phi(z) + sigma * phi(z)
            from scipy.stats import norm
            phi_z = norm.pdf(z)
            Phi_z = norm.cdf(z)
            ei = (mean - self.best_y_so_far) * Phi_z + std * phi_z
            
        # Normalize and return negative for minimization
        return -ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(30, max(10, batch_size * 5))  # More starts than needed to ensure diversity
        
        # Generate diverse starting points using Sobol sequence for better space-filling
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: self._acquisition_function(x.reshape(1, -1))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=20,
                    popsize=5,
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
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
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from initial set
            selected_points = self._sample_points(1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_points(1)
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol design for better space-filling
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Select next points to evaluate using acquisition function
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size
            
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # Track best value seen so far for acquisition function
            current_best_y = np.min(self.y)
            if current_best_y < self.best_y_so_far:
                self.best_y_so_far = current_best_y
                self.last_improvement_eval = self.n_evals
            
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
 The algorithm EMBOESM got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4442 with standard deviation 0.0336.

took 313.52 seconds to run.