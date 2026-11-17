from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EnsembleThompsonSamplingSobolBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.batch_size = 2  # Parallel evaluation
        self.n_init = max(10, 2 * dim)  # More points for ensemble training

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Latin Hypercube candidates within specified bounds."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_models(self, X, y):
        """Fit GP and Random Forest models to the data."""
        # GP with Matérn kernel
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,  # Reduced for speed
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp_model.fit(X, y)

        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=20,
            max_depth=5,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        rf_model.fit(X, y)

        return gp_model, rf_model

    def _acquisition_function(self, X, gp_model, rf_model):
        """Hybrid Thompson Sampling: sample GP + use RF predictions."""
        gp_mean, gp_std = gp_model.predict(X, return_std=True)
        rf_pred = rf_model.predict(X)
        # Sample function from GP posterior
        gp_sample = gp_mean + gp_std * self.rng.normal(size=len(gp_mean))
        # Combine GP sample and RF prediction
        combined = 0.7 * gp_sample + 0.3 * rf_pred  # Weighted average
        return combined

    def _select_next_points(self):
        """Generate candidates and select batch with Thompson Sampling."""
        candidates = self._sample_points(100)  # 100 Sobol candidates
        gp_model, rf_model = self.models
        acq_vals = self._acquisition_function(candidates, gp_model, rf_model)
        # Select top batch_size candidates with max acquisition values
        top_indices = np.argsort(acq_vals)[-self.batch_size:]
        return candidates[top_indices]

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
        # Initial evaluation with Latin Hypercube
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial models
        self.models = self._fit_models(self.X, self.y)

        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Refit models with updated data
            self.models = self._fit_models(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
