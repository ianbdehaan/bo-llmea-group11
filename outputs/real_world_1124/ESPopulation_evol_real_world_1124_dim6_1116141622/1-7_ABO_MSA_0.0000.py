from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

class ABO_MSA:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, budget // 4))  # Adaptive initial points based on budget
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model with dynamic hyperparameters
        self.gp_model = None
        
        # Acquisition function parameters
        self.noise_var = 1e-6  # Fixed noise variance to avoid expensive optimization
        self.exploration_factor = 0.01  # Adaptive exploration factor
        self.convergence_threshold = 1e-4  # For early termination
        self.convergence_window = 5  # Number of evaluations to check for convergence
        
    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube Sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with dynamic hyperparameters for better accuracy
        if self.gp_model is None:
            # Use RBF kernel with automatic relevance detection
            # For first few evaluations, use fixed length scales to avoid overfitting
            if len(y) < 10:
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            else:
                # Use more flexible kernel for better fit when more data available
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_var,
                n_restarts_optimizer=min(2, max(0, len(y) // 5)),  # Dynamic restarts based on data size
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
        
        # Fit the model with new data
        try:
            self.gp_model.fit(X, y)
        except Exception as e:
            # If fitting fails, use a simpler model
            warnings.warn(f"GP fitting failed: {e}, using fallback")
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_var,
                n_restarts_optimizer=0,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            self.gp_model.fit(X, y)
            
        return self.gp_model

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function with improved exploration
        if self.y is None or len(self.y) == 0:
            # If no data yet, use a simple heuristic
            return np.ones((X.shape[0], 1)) * 0.1
        
        # Get GP predictions
        try:
            mu, sigma = self.gp_model.predict(X, return_std=True)
        except Exception:
            # Fallback to simpler computation if prediction fails
            return np.ones((X.shape[0], 1)) * 0.01
            
        # Best observed value so far (for EI calculation)
        y_best = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Calculate Expected Improvement with adaptive exploration factor
        improvement = y_best - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add adaptive exploration factor to encourage diversity
        # Increase exploration when we have fewer samples
        if len(self.y) < 10:
            ei *= (1.0 + self.exploration_factor * 2.0)  # More exploration early
        elif len(self.y) < 20:
            ei *= (1.0 + self.exploration_factor * 1.5)  # Moderate exploration
        else:
            ei *= (1.0 + self.exploration_factor * 0.5)  # Less exploration later
            
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start acquisition optimization for better exploration
        best_acq_value = -np.inf
        best_x = None
        
        # Try multiple random starts with different sampling strategies
        n_starts = min(5, max(1, batch_size * 2))  # Limit starts to avoid overhead
        
        for _ in range(n_starts):
            # Determine sampling strategy based on iteration
            strategy_idx = _ % 3
            
            if strategy_idx == 0:
                # Random sampling
                candidates = self._sample_points(min(1000, max(100, self.budget * 2)))
            elif strategy_idx == 1:
                # Sobol sequence sampling
                sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
                n_samples = min(1000, max(100, self.budget * 2))
                samples = sampler.random(n=n_samples)
                candidates = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
            else:
                # LHS sampling
                candidates = self._sample_points(min(1000, max(100, self.budget * 2)))
            
            # Evaluate acquisition function for all candidates
            acq_values = self._acquisition_function(candidates)
            
            # Select the best point among candidates
            if len(acq_values) > 0:
                max_idx = np.argmax(acq_values)
                if acq_values[max_idx] > best_acq_value:
                    best_acq_value = acq_values[max_idx]
                    best_x = candidates[max_idx].reshape(1, -1)
        
        # Return the best point found
        if best_x is not None:
            return np.array(best_x)
        else:
            # Fallback to random sampling if nothing works
            return self._sample_points(batch_size)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X using function evaluation
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y = []
        for x in X:
            try:
                y.append(func(x))
            except Exception:
                # Handle function evaluation errors
                y.append(np.inf)  # Assign worst possible value
                
        self.n_evals += len(X)
        return np.array(y).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _check_convergence(self):
        # Check for convergence to avoid unnecessary evaluations
        if len(self.y) < self.convergence_window + 1:
            return False
            
        recent_values = self.y[-self.convergence_window:]
        std_dev = np.std(recent_values)
        mean_val = np.mean(recent_values)
        
        # If standard deviation is very small compared to mean, consider converged
        if std_dev < self.convergence_threshold * abs(mean_val) and std_dev < 1e-6:
            return True
        return False

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with initial samples
        if self.X is None:
            X_init = self._sample_points(self.n_init)
            y_init = self._evaluate_points(func, X_init)
            self._update_eval_points(X_init, y_init)
        
        # Fit the model on initial data
        self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Check for convergence
            if self._check_convergence():
                break
                
            # Fit model with current data
            try:
                self._fit_model(self.X, self.y)
            except Exception:
                # Fallback to simple sampling if model fitting fails
                break
            
            # Adaptive batch size based on remaining budget and iterations
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 10:
                batch_size = 1
            elif remaining_budget <= 50:
                batch_size = min(3, max(1, remaining_budget // 10))
            else:
                batch_size = min(5, max(1, (self.budget - self.n_evals) // 10))
            
            # Select next points using acquisition function
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            
            # Update the data
            self._update_eval_points(next_X, next_y)
            
            # Early termination if budget is almost exhausted
            if self.n_evals >= self.budget:
                break
                
        # Return best solution found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
