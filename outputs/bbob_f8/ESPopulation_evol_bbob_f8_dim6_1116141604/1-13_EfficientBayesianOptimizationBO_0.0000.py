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

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process model with Matern kernel
        try:
            # Filter out invalid values before fitting
            valid_mask = np.isfinite(y.flatten())
            if not np.any(valid_mask):
                raise ValueError("No valid y values for GP fitting")
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Check if we have enough points
            if len(y_valid) < 2:
                # If too few points, use a simpler approach or return early
                kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=1,
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
            else:
                # Use a more stable approach by setting noise_level explicitly
                kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
                
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=min(3, len(y_valid) - 1),
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
            
            # Fit the model with valid data
            gp.fit(X_valid, y_valid.flatten())
            return gp
            
        except Exception as e:
            # Fallback to a simpler model if complex fitting fails
            warnings.warn(f"GP fitting failed with error: {e}, using simple model")
            try:
                kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=1,
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                if len(y) >= 2:
                    gp.fit(X, y.flatten())
                else:
                    # If we have no valid data points, return a basic model
                    warnings.warn("Returning basic model due to insufficient data")
                    return None
                return gp
            except Exception as fallback_error:
                warnings.warn(f"Fallback GP fitting also failed: {fallback_error}")
                return None

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function
        if self.model is None:
            # Return zeros if no model yet
            return np.zeros((X.shape[0], 1))
        
        try:
            # Get predictions from GP model
            mean, std = self.model.predict(X, return_std=True)
            
            # Ensure std is positive and not too small
            std = np.maximum(std, self.eps)
            
            # Get the best observed value so far (minimum for minimization)
            y_best = np.min(self.y) if len(self.y) > 0 else 0
            
            # Calculate EI with numerical stability
            improvement = y_best - mean
            # Avoid division by zero and negative std
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Add small epsilon to prevent numerical issues
                ei = improvement * norm.cdf(improvement / (std + self.eps)) + std * norm.pdf(improvement / (std + self.eps))
            
            # Clip values to avoid extreme outliers
            ei = np.clip(ei, -1e10, 1e10)
            
            # Return as column vector
            return ei.reshape(-1, 1)
        except Exception:
            # Return zeros if calculation fails for any reason
            return np.zeros((X.shape[0], 1))

    def _select_next_points(self, batch_size):
        # Use a hybrid approach for acquisition function optimization:
        # 1. First try local optimization with L-BFGS 
        # 2. Then use global sampling if needed
        
        candidates = []
        
        # For the first few iterations, use more diverse sampling
        if len(self.X) < self.n_init * 0.5 or batch_size == 1:
            # Use random sampling for diversity
            candidates = self._sample_points(batch_size)
        else:
            # Use a hybrid approach with both local and global search
            
            # Try local optimization with multiple starting points but limit attempts
            n_starts = min(10, max(3, batch_size * 3))
            
            # Sample initial points using LHS
            x0_candidates = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_x = None
            
            for i in range(min(5, n_starts)):  # Limit local optimization attempts
                try:
                    # Use bounds for constraint handling
                    bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
                    
                    # Local optimization using L-BFGS
                    res = minimize(
                        lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                        x0=x0_candidates[i],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 20}
                    )
                    
                    if res.success:
                        candidate_x = res.x.reshape(1, -1)
                        acq_val = self._acquisition_function(candidate_x)[0, 0]
                        
                        if acq_val > best_acq_val:
                            best_acq_val = acq_val
                            best_x = candidate_x
                        
                except Exception:
                    continue
            
            # If we have a good candidate from local search, use it
            if best_x is not None:
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
                # Check if result is valid
                if np.isfinite(y_val):
                    y_values.append(y_val)
                else:
                    # Assign a large value if function evaluation fails or returns invalid result
                    y_values.append(1e10)
            except Exception:
                # If function evaluation fails, assign a large value
                y_values.append(1e10)
        
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
        # Filter out any invalid results
        valid_mask = np.isfinite(self.y.flatten())
        if not np.any(valid_mask):
            # If no valid points, return worst possible result
            return 1e10, np.zeros(self.dim)
            
        y_valid = self.y[valid_mask]
        X_valid = self.X[valid_mask]
        
        best_idx = np.argmin(y_valid)
        best_y = float(y_valid[best_idx])
        best_x = X_valid[best_idx].copy()
        
        return best_y, best_x
