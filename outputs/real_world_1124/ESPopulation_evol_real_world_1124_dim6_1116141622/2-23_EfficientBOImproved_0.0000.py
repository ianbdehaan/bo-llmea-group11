from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

class EfficientBOImproved:
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
        
        # GP model with fixed hyperparameters for efficiency
        self.gp_model = None
        
        # Acquisition function parameters
        self.noise_var = 1e-6  # Fixed noise variance to avoid expensive optimization
        self.exploration_factor = 0.01  # Adaptive exploration factor
        self.acq_switch_threshold = 0.3  # Threshold for switching between EI and UQ acquisition functions
        self.progress = 0.0  # Optimization progress indicator

    def _sample_points(self, n_points):
        # Sample points using a hybrid approach: LHS for initial sampling, Sobol for diversity
        if n_points <= 10:
            # Use LHS for small samples to ensure good coverage
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        else:
            # Use Sobol sequence for larger samples to ensure better distribution
            sampler = qmc.Sobol(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_n_points)
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with adaptive hyperparameter optimization for efficiency
        if self.gp_model is None:
            # Use a simple RBF kernel with automatic relevance detection
            # Fixed length scales to avoid expensive optimization
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_var,
                n_restarts_optimizer=3,  # Limited restarts for efficiency
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
        
        # Fit the model with new data
        self.gp_model.fit(X, y)
        return self.gp_model

    def _acquisition_function(self, X):
        # Adaptive acquisition function that switches between EI and UQ based on progress
        if self.y is None or len(self.y) == 0:
            # If no data yet, use a simple heuristic
            return np.ones((X.shape[0], 1)) * 0.1
        
        # Get GP predictions
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Best observed value so far (for EI calculation)
        y_best = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Switch between acquisition functions based on progress
        if self.progress < self.acq_switch_threshold:
            # Early stage: use Expected Improvement for exploration
            improvement = y_best - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            acq_values = ei.reshape(-1, 1)
        else:
            # Later stage: use Uncertainty Quantification for exploitation
            # Using the standard deviation as uncertainty measure
            uq = sigma.reshape(-1, 1)
            acq_values = uq
        
        # Add exploration factor to encourage diversity
        return acq_values

    def _select_next_points(self, batch_size):
        # Multi-start acquisition optimization with early stopping and diversity consideration
        best_acq_value = -np.inf
        best_x = None
        
        # Track best points across different sampling strategies
        all_candidates = []
        all_acq_values = []
        
        # Try multiple random starts with different sampling strategies
        n_starts = min(5, max(1, batch_size * 2))  # Limit starts to avoid overhead
        
        for start_idx in range(n_starts):
            # Sample points using different strategies based on iteration
            if start_idx < 2:
                # Use Sobol sequence for first few starts
                n_samples = min(1000, max(100, self.budget * 3))
                candidates = self._sample_points(n_samples)
            elif start_idx < 4:
                # Use random sampling for middle starts
                candidates = self.bounds[0] + self.rng.random((n_samples, self.dim)) * (self.bounds[1] - self.bounds[0])
            else:
                # Use LHS for final starts
                n_samples = min(1000, max(100, self.budget * 3))
                candidates = self._sample_points(n_samples)
            
            # Evaluate acquisition function for all candidates
            acq_values = self._acquisition_function(candidates)
            
            # Store candidates and their values
            all_candidates.append(candidates)
            all_acq_values.append(acq_values.flatten())
        
        # Select the best points considering diversity and acquisition value
        all_candidates = np.vstack(all_candidates)
        all_acq_values = np.hstack(all_acq_values)
        
        # Get indices of top candidates
        top_indices = np.argsort(all_acq_values)[-batch_size:]
        
        # Return the best points found
        return all_candidates[top_indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X using function evaluation
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y = []
        for x in X:
            y.append(func(x))
            
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with initial samples
        if self.X is None:
            # Hybrid initialization: LHS for first few points, then Sobol
            n_init_lh = min(10, self.n_init)
            X_init_lh = self._sample_points(n_init_lh)
            y_init_lh = self._evaluate_points(func, X_init_lh)
            
            if self.n_init > 10:
                # Use Sobol for remaining points
                n_init_sobol = self.n_init - n_init_lh
                X_init_sobol = self._sample_points(n_init_sobol)
                y_init_sobol = self._evaluate_points(func, X_init_sobol)
                
                X_init = np.vstack([X_init_lh, X_init_sobol])
                y_init = np.vstack([y_init_lh, y_init_sobol])
            else:
                X_init = X_init_lh
                y_init = y_init_lh
                
            self._update_eval_points(X_init, y_init)
        
        # Fit the model on initial data
        self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit model with current data
            try:
                self._fit_model(self.X, self.y)
            except Exception:
                # Fallback to simple sampling if model fitting fails
                break
            
            # Update optimization progress
            self.progress = min(1.0, self.n_evals / self.budget)
            
            # Select next points using acquisition function
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, max(1, remaining_budget // 4))
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
