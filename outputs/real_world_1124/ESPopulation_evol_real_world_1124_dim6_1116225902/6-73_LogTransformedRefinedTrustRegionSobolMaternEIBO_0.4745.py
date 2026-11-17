from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class LogTransformedRefinedTrustRegionSobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.original_y: np.ndarray = None  # Original (untransformed) y values
        self.y: np.ndarray = None  # Transformed y values for GP modeling
        self.n_evals = 0
        self.n_init = max(5, 3 * dim)  # Increase initial design to 3d
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        # Trust region parameters
        self.tr_center = None
        self.tr_radius = 1.0  # Smaller initial radius
        self.max_tr_radius = 5.0  # Half of domain range
        self.min_tr_radius = 0.1  # Minimum allowable radius
        self.iteration = 0  # Track iterations for refitting frequency

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Sobol candidates within specified bounds (default: overall domain)."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Gaussian Process with Matern kernel and ARD, fewer restarts for speed."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,  # Increased restarts for better hyperparameter fitting
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle zero variance
        return ei

    def _select_next_points(self):
        """Generate candidates within current trust region, select top 2."""
        if self.tr_center is None:
            return self._sample_points(50)  # Fallback to global search if no trust region
        
        # Compute trust region bounds (clamped to domain)
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        
        candidates = self._sample_points(100, low=low, high=high)  # 100 candidates in trust region
        acq_vals = self._acquisition_function(candidates)
        # Select top 2 candidates with highest EI
        top_indices = np.argsort(acq_vals, axis=0)[-2:][::-1]
        next_X = candidates[top_indices].copy()
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        original_y = np.zeros((n_points, 1))
        transformed_y = np.zeros((n_points, 1))
        for i in range(n_points):
            original_value = func(X[i])
            original_y[i] = original_value
            # Apply log transformation to stabilize GP fit
            transformed_value = np.log(original_value + 1e-6)  # Avoid log(0)
            transformed_y[i] = transformed_value
            self.n_evals += 1
        return transformed_y, original_y

    def _update_eval_points(self, new_X, new_transformed_y, new_original_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_transformed_y
            self.original_y = new_original_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_transformed_y])
            self.original_y = np.vstack([self.original_y, new_original_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Sobol
        X_init = self._sample_points(self.n_init)
        transformed_y_init, original_y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, transformed_y_init, original_y_init)
        
        # Set initial trust region center
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_transformed_y, next_original_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_transformed_y, next_original_y)
            
            # Update trust region
            current_best_idx = np.argmin(self.y)
            new_best = next_transformed_y[0] < self.y[best_idx]
            
            if new_best:
                # Expand trust region when progress is made
                self.tr_radius = min(self.tr_radius * 1.1, self.max_tr_radius)
            else:
                # Contract when no improvement
                self.tr_radius = max(self.tr_radius * 0.8, self.min_tr_radius)
            
            # Update best solution and trust region center
            best_idx = current_best_idx
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model every 2 iterations to reduce overhead
            self.iteration += 1
            if self.iteration % 2 == 0:
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.original_y[best_idx]  # Return original y
        best_x = self.X[best_idx]
        return best_y, best_x
