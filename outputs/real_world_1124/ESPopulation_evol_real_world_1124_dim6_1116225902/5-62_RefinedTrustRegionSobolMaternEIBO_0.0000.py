from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class RefinedTrustRegionSobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1  # More diverse initial samples
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        # Trust region parameters
        self.tr_center = None
        self.tr_radius = 1.0  # Smaller initial radius
        self.max_tr_radius = 5.0
        self.min_tr_radius = 0.1

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Sobol candidates with random perturbation for diversity."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        perturbation = self.rng.uniform(-0.05, 0.05, scaled_samples.shape)  # Small random perturbation
        return scaled_samples + perturbation

    def _fit_model(self, X, y):
        """Fit Gaussian Process with Matérn kernel and ARD, lightweight refitting."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,  # Reduced restarts for speed
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization with noise handling."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / (std + 1e-8)  # Avoid division by zero
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0
        return ei

    def _select_next_points(self):
        """Generate 100 candidates, select top 2 with diversity."""
        if self.tr_center is None:
            return self._sample_points(50)
        
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        candidates = self._sample_points(100, low=low, high=high)
        acq_vals = self._acquisition_function(candidates)
        
        # Select top 2 most diverse candidates
        indices = np.argsort(acq_vals)[-2:]
        next_X = candidates[indices]
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with enhanced Sobol design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Set initial trust region center
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update trust region
            current_best_idx = np.argmin(self.y)
            improved = next_y[0] < self.y[best_idx] or (len(next_y) > 1 and next_y[1] < self.y[best_idx])
            
            if improved:
                self.tr_radius = min(self.tr_radius * 1.1, self.max_tr_radius)
            else:
                self.tr_radius = max(self.tr_radius * 0.8, self.min_tr_radius)
            
            best_idx = current_best_idx
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
