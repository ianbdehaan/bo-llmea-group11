from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
import warnings

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
        self.n_init = max(2 * dim, 10)  # Initial points for Bayesian optimization
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model = None
        self.noise_level = 1e-6  # Small noise level to avoid numerical issues
        
        # Batch size that decreases with budget consumption
        self.batch_size = min(max(1, int(np.ceil(self.budget / 50))), 5)
        
        # Acquisition function parameters
        self.eps = 1e-8
        self.large_value = 1e10  # Large value to replace invalid evaluations

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process model with Matern kernel
        # Handle invalid values in y
        valid_indices = np.isfinite(y.flatten())
        if not np.any(valid_indices):
            # All values are invalid, return a simple model
            kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                n_restarts_optimizer=1,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            # Use a constant prediction
            gp.fit(X[:1], np.array([0.0]))
            return gp
            
        # Filter out invalid values
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]
        
        try:
            # Use a more stable approach by setting noise_level explicitly
            kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
            
            # Fit model with noise level set to a small value
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                n_restarts_optimizer=min(3, max(1, len(y_valid) // 2)),
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            # Fit the model
            gp.fit(X_valid, y_valid.flatten())
            return gp
        except Exception as e:
            # Fallback to a simpler model if complex fitting fails
            warnings.warn(f"GP fitting failed with error: {e}, using simple model")
            kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                n_restarts_optimizer=1,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            # If we have valid points, fit with them; otherwise use simple model
            if len(y_valid) > 0:
                gp.fit(X_valid, y_valid.flatten())
            else:
                # Use one point for fallback
                gp.fit(X[:1], np.array([0.0]))
            return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function
        if self.model is None or len(self.X) == 0:
            # Return zeros if no model yet
            return np.zeros((X.shape[0], 1))
        
        # Get predictions from GP model
        try:
            mean, std = self.model.predict(X, return_std=True)
        except Exception:
            # If prediction fails, return zeros
            return np.zeros((X.shape[0], 1))
            
        # Get the best observed value so far (minimum for minimization)
        y_best = np.min(self.y) if len(self.y) > 0 else 0
        
        # Calculate EI
        improvement = y_best - mean
        # Avoid division by zero and negative std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Clip std to avoid numerical issues
            std_safe = np.clip(std, self.eps, None)
            ei = improvement * norm.cdf(improvement / std_safe) + std_safe * norm.pdf(improvement / std_safe)
        
        # Return as column vector
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Use a hybrid approach for acquisition function optimization:
        # 1. First try local optimization with L-BFGS 
        # 2. Then use global sampling if needed
        
        candidates = []
        
        # For the first few iterations, use more diverse sampling
        if len(self.X) < self.n_init * 0.5 or batch_size == 1:
            # Use a combination of random and L-BFGS optimization
            
            # Generate initial candidate points using Latin Hypercube Sampling
            init_points = self._sample_points(100)
            
            # Evaluate acquisition function for these points
            acq_values = self._acquisition_function(init_points)
            
            # Select best points from the candidates
            valid_indices = np.isfinite(acq_values.flatten())
            if np.any(valid_indices):
                best_indices = np.argsort(acq_values.flatten()[valid_indices])[-batch_size:][::-1]
                candidates = init_points[valid_indices][best_indices]
            else:
                # Fallback to random points
                candidates = self._sample_points(batch_size)
        else:
            # Use a hybrid approach with both local and global search
            
            # Try local optimization with multiple starting points
            n_starts = min(20, max(5, batch_size * 5))
            
            # Sample initial points using LHS
            x0_candidates = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_x = None
            
            for i in range(min(10, n_starts)):  # Limit local optimization attempts
                try:
                    # Use bounds for constraint handling
                    bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
                    
                    # Local optimization using L-BFGS
                    res = minimize(
                        lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                        x0=x0_candidates[i],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50}
                    )
                    
                    if res.success:
                        candidate_x = res.x.reshape(1, -1)
                        acq_val = self._acquisition_function(candidate_x)[0, 0]
                        
                        if np.isfinite(acq_val) and acq_val > best_acq_val:
                            best_acq_val = acq_val
                            best_x = candidate_x
                        
                except Exception:
                    continue
            
            # If we have a good candidate from local search, use it
            if best_x is not None and np.isfinite(best_x).all():
                candidates = best_x
            else:
                # Fallback to random sampling for diversity
                candidates = self._sample_points(batch_size)
        
        return candidates

    def _evaluate_points(self, func, X):
        # Evaluate the points in X using the provided function
        # func: takes array of shape (n_dims,) and returns np.float64.
        # Return array of shape (n_points, 1)
        y_values = []
        for x in X:
            try:
                y_val = func(x)
                # Check if the value is valid
                if np.isfinite(y_val):
                    y_values.append(y_val)
                else:
                    # Replace invalid values with large finite number
                    y_values.append(self.large_value)
            except Exception:
                # If function evaluation fails, assign a large value
                y_values.append(self.large_value)
        
        return np.array(y_values).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y with new evaluations
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
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, init_points)
        
        # Update evaluation points
        self._update_eval_points(init_points, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running out of budget, reduce batch size
            current_batch_size = min(self.batch_size, self.budget - self.n_evals)
            
            if current_batch_size <= 0:
                break
                
            # Select next points to evaluate using acquisition function
            next_points = self._select_next_points(current_batch_size)
            
            # Evaluate the selected points
            y_next = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, y_next)
            
            # Update model with new data (use selective retraining for efficiency)
            if len(self.X) % 5 == 0 or len(self.X) < 20:
                # Retrain model every few iterations or when we have few points
                self.model = self._fit_model(self.X, self.y)
            
            # Check if budget is nearly exhausted to adjust batch size
            if self.n_evals + current_batch_size > self.budget * 0.95:
                # Reduce batch size to avoid exceeding budget
                current_batch_size = min(current_batch_size, self.budget - self.n_evals)
                
        # Return best found solution
        valid_indices = np.isfinite(self.y.flatten())
        if not np.any(valid_indices):
            # If no valid points found, return the first point
            best_idx = 0
        else:
            # Find best among valid points
            valid_y = self.y[valid_indices]
            best_idx = np.argmin(valid_y)
            # Adjust index to original array
            best_idx = np.where(valid_indices)[0][best_idx]
            
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
