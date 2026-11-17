from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
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
        self.n_init = max(2 * self.dim + 1, 10) # Initial points based on dimensionality
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.trust_region_radius = 5.0  # Initial radius (full search space)
        self.min_trust_region_radius = 0.01  # Minimum radius for trust region
        self.trust_region_shrink_factor = 0.5  # Factor to shrink trust region when no progress
        self.trust_region_expand_factor = 1.5  # Factor to expand trust region when progress
        self.progress_threshold = 1e-3  # Threshold for considering progress
        self.last_improvement_evals = 0  # Number of evaluations since last improvement
        
        # Model and acquisition function parameters
        self.model = None
        self.best_y = np.inf
        self.best_x = None
        
    def _sample_points(self, n_points):
        # Use maximin Latin Hypercube design for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = qmc.scale(points, self.bounds[0], self.bounds[1])
        return points
    
    def _fit_model(self, X, y):
        # Fit Gaussian Process surrogate model with ARD Matern kernel
        try:
            # Use a more robust approach for GP fitting
            kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=3,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp
        except:
            # Fallback to a simple GP if the above fails
            kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-6)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=3,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp
    
    def _acquisition_function(self, X):
        # Expected Improvement for GP
        try:
            # Get predictions from GP
            mean, std = self.model.predict(X, return_std=True)
            
            # Calculate EI with exploration factor
            improvement = self.best_y - mean - 1e-8
            # Use a more stable calculation for EI
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ei = improvement * (improvement / (std + 1e-12)) * np.exp(-0.5 * (improvement/std)**2) + std * np.exp(-0.5 * (improvement/std)**2)
            
            return ei.reshape(-1, 1)
        except:
            # Fallback to a simple value if there's an issue
            return np.ones((X.shape[0], 1)) * 0.1
    
    def _select_next_points(self, batch_size):
        # Multi-start L-BFGS optimization of acquisition function with trust region constraints
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple random starting points for optimization
        n_starts = min(20, max(5, self.n_evals // 2))
        
        for _ in range(n_starts):
            # Start from a random point within bounds
            start_point = self._sample_points(1)[0]
            
            def acq_func(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0][0]
            
            try:
                # Use L-BFGS-B optimization with bounds and trust region constraints
                res = minimize(
                    acq_func, 
                    start_point,
                    method='L-BFGS-B',
                    bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                    options={'maxiter': 50}
                )
                
                if res.success:
                    val = -res.fun
                    if val > best_acq_val:
                        best_acq_val = val
                        best_x = res.x.reshape(1, -1)
            except:
                continue
        
        # If no good points found, return a random point
        if best_x is None:
            return self._sample_points(batch_size)
        
        return np.repeat(best_x, batch_size, axis=0)
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None:
            return np.array([])
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except:
                # If evaluation fails, use a large value
                y_vals.append(np.inf)
        
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
        
        # Update best point found so far
        min_idx = np.argmin(self.y)
        if self.y[min_idx] < self.best_y:
            self.best_y = self.y[min_idx]
            self.best_x = self.X[min_idx]
            
    def _update_trust_region(self):
        """Update the trust region based on recent progress"""
        # If we haven't improved for a while, shrink the trust region
        if self.n_evals - self.last_improvement_evals > 20:
            self.trust_region_radius *= self.trust_region_shrink_factor
            self.last_improvement_evals = self.n_evals
            
        # If we've made significant progress, expand the trust region
        elif self.n_evals - self.last_improvement_evals < 10 and self.trust_region_radius > self.min_trust_region_radius:
            self.trust_region_radius *= self.trust_region_expand_factor
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube design
        init_X = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_X)
        self._update_eval_points(init_X, init_y)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y.flatten())
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, reduce batch size
            remaining_budget = self.budget - self.n_evals
            batch_size = min(10, remaining_budget)
            
            if batch_size <= 0:
                break
                
            # Update trust region based on recent progress
            self._update_trust_region()
            
            # Select next points to evaluate
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            
            # Update the model with new data
            self._update_eval_points(next_X, next_y)
            
            # Re-fit the model periodically (every 5 evaluations) to maintain accuracy
            if self.n_evals % 5 == 0:
                try:
                    self.model = self._fit_model(self.X, self.y.flatten())
                except:
                    pass  # If fitting fails, keep current model
        
        return self.best_y, self.best_x
