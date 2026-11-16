from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
import warnings

class EMBO_SB_TRE:
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
        self.trust_region_radius = 2.0
        self.min_trust_region_radius = 0.1
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.8
        self.trust_region_expand_factor = 1.2
        self.progress_threshold = 0.01
        
        # Acquisition function parameters
        self.noise_level = 1e-6  # Small noise for numerical stability
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process regressor with ARD Matern kernel
        # Use only a few hyperparameter restarts for efficiency
        try:
            # Use a sparse GP with limited restarts to maintain efficiency
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            
            # Fit GP with limited optimization restarts for efficiency
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                n_restarts_optimizer=3,  # Limit restarts for computational efficiency
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X, y.flatten())
            return gp
        except Exception:
            # Fallback to a simpler model if complex fitting fails
            kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                n_restarts_optimizer=1,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y.flatten())
            return gp

    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration weight
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP (mean and std)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current best value
        if self.y is not None:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Avoid division by zero or negative standard deviation
        std = np.maximum(std, 1e-8)
        
        # Calculate expected improvement
        improvement = best_y - mean
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = -np.inf
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
                    
                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization"""
        selected_points = []
        
        # First select one point using acquisition optimization
        best_point = self._optimize_acquisition(n_starts=20)
        if best_point is not None:
            selected_points.append(best_point)
        else:
            # Fallback to random point
            selected_points.append(self._sample_points(1))
        
        # Add additional points based on trust region around current best
        if len(selected_points) < batch_size and self.X is not None and len(self.X) > 0:
            # Get the best point so far
            best_idx = np.argmin(self.y)
            best_point = self.X[best_idx]
            
            # Select additional points in trust region around best point
            remaining = batch_size - len(selected_points)
            for _ in range(remaining):
                # Sample within trust region
                sample_point = best_point + self.rng.uniform(-self.trust_region_radius, 
                                                              self.trust_region_radius, self.dim)
                # Clip to bounds
                sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                selected_points.append(sample_point.reshape(1, -1))
        
        return np.vstack(selected_points)

    def _update_trust_region(self, improvement):
        """Update trust region based on recent progress"""
        if improvement < self.progress_threshold:
            # Shrink trust region if little progress
            self.trust_region_radius = max(
                self.min_trust_region_radius,
                self.trust_region_radius * self.trust_region_shrink_factor
            )
        else:
            # Expand trust region if good progress
            self.trust_region_radius = min(
                self.max_trust_region_radius,
                self.trust_region_radius * self.trust_region_expand_factor
            )

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
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Keep track of the best improvement for trust region adjustment
        prev_best = np.min(self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
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
            
            # Update trust region based on improvement
            current_best = np.min(self.y)
            improvement = prev_best - current_best
            self._update_trust_region(improvement)
            prev_best = current_best
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
