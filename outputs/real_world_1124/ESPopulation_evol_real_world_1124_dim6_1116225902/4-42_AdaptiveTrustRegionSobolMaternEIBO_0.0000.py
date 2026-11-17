from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class AdaptiveTrustRegionSobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim + 3)  # Improved initial design for 6D problem
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        # Trust region parameters
        self.tr_center = None
        self.tr_radius = 1.0  # Smaller initial radius
        self.max_tr_radius = 5.0  # Half of domain range
        self.min_tr_radius = 0.1  # Minimum allowable radius
        self.refit_interval = 1  # Start with frequent model refits

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
        """Fit Gaussian Process with Matérn kernel and ARD."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        n_restarts = 3 if self.refit_interval == 1 else 1  # Fewer restarts when refitting less frequently
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization with diversity promotion."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle zero variance
        return ei

    def _select_next_points(self):
        """Generate candidates within current trust region with diversity promotion."""
        if self.tr_center is None:
            return self._sample_points(50)  # Fallback to global search if no trust region
        
        # Compute trust region bounds (clamped to domain)
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        
        # Generate candidates with diversity mix: 95% Sobol, 5% random
        sobol_candidates = self._sample_points(95, low=low, high=high)
        random_candidates = self._sample_points(5, low=low, high=high)
        candidates = np.vstack((sobol_candidates, random_candidates))
        
        acq_vals = self._acquisition_function(candidates)
        top_indices = np.argsort(acq_vals)[-2:]  # Select top 2 acquisition values
        next_X = candidates[top_indices]
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
        # Initial evaluation with Sobol
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
            improved = next_y[0] < self.y[best_idx]
            
            if improved:
                # Expand trust region when progress is made
                self.tr_radius = min(self.tr_radius * 1.1, self.max_tr_radius)
                # Make model refits more frequent when making progress
                self.refit_interval = 1
            else:
                # Contract when no improvement
                self.tr_radius = max(self.tr_radius * 0.95, self.min_tr_radius)
                # Reduce model refits when progress stalls
                self.refit_interval = 2 if self.refit_interval > 1 else 1
            
            # Update best solution and trust region center
            best_idx = current_best_idx
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model with updated data
            if self.n_evals % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
