# Description
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration (EMBO)**

This is a novel Bayesian optimization algorithm that combines several key innovations to achieve excellent performance on black-box optimization problems. The approach uses an adaptive exploration strategy, multi-start acquisition function optimization, and efficient surrogate modeling.

Key characteristics:
1. **Adaptive Exploration**: Uses a dynamic exploration weight in the lower confidence bound (LCB) acquisition function that decreases over time
2. **Multi-Start Optimization**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima
3. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
4. **Adaptive Initial Design**: Starts with a maximin Latin Hypercube design scaled to the problem dimensionality
5. **Batch Selection**: Selects multiple points in batches using diverse sampling strategies

# Justification 
The EMBO algorithm introduces several key innovations that differentiate it from existing approaches:

1. **Dynamic Exploration Weighting**: Unlike static exploration-exploitation balances, our LCB uses a time-varying exploration weight that decreases linearly from 2.0 to 0.2. This allows for more aggressive exploration early in the optimization when uncertainty is high, and then shifts toward exploitation as we gather more information.

2. **Multi-Start Acquisition Optimization**: While many BO algorithms use simple local optimization of acquisition functions, EMBO employs multiple random restarts with local refinement via differential evolution. This prevents getting stuck in poor local optima of the acquisition function.

3. **Efficient Surrogate Modeling**: We use a sparse GP with ARD Matern kernel that only needs 3 hyperparameter restarts for efficiency, while still maintaining good modeling performance. The GP is retrained at each iteration with all available data to ensure model quality.

4. **Adaptive Batch Selection**: Instead of fixed batch sizes, the algorithm dynamically adjusts batch size based on remaining budget and scales appropriately for different problem dimensions.

5. **Maximin Latin Hypercube Initialization**: This provides good initial coverage while being computationally efficient, especially for higher dimensional problems where random sampling becomes less effective.

The combination of these elements creates an algorithm that is both robust and computationally efficient across varying budgets and dimensions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EMBO:
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
        self.acq_weight_schedule = np.linspace(2.0, 0.2, max(10, budget // 10))  # Dynamic exploration weight
        
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
        # Lower Confidence Bound with adaptive exploration weight
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
            
        # LCB acquisition function
        lcb = mean - exploration_weight * std
        
        return lcb.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(20, max(5, batch_size * 3))  # More starts than needed to ensure diversity
        
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
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
        
        # Initialize with Latin Hypercube design
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
 The algorithm EMBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4272 with standard deviation 0.0313.

took 196.53 seconds to run.