from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustMaternLogEIAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model
        self.tr_bounds = self.bounds.copy()  # Trust region bounds
        self.best_point = None
        self.progress_threshold = 5  # No improvement threshold
        self.progress_count = 0

    def _sample_points(self, n_points, bounds=None):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        if bounds is None:
            bounds = self.tr_bounds
        samples = sobol.random(n=n_points)
        return bounds[0] + (bounds[1] - bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        best_y = np.min(self.y)
        imp = best_y - mean
        imp = np.where(imp < 0, 0, imp)
        Z = imp / (std + 1e-8)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        log_ei = np.log(ei + 1e-10)
        return log_ei

    def _select_next_points(self, batch_size):
        # Generate candidates within trust region
        candidates = self._sample_points(n_points=100)
        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmax(acq_vals)
        best_candidate = candidates[best_idx]

        # Optimize acquisition function within trust region
        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],
            x0=best_candidate,
            bounds=[(self.tr_bounds[0][i], self.tr_bounds[1][i]) for i in range(self.dim)],
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        return result.x[np.newaxis, :]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def _update_tr_bounds(self, new_point):
        current_best = self.best_point
        new_best = self.X[np.argmin(self.y)] if self.best_point is None or self.y.min() < self.best_point[1] else current_best
        if self.best_point is None or new_best[1] < self.best_point[1]:
            self.best_point = (new_best[0], new_best[1])  # (point, value)
            self.progress_count = 0
            return  # Reset progress counter on improvement
        self.progress_count += 1

        # Shrink trust region if progress stalls
        if self.progress_count >= self.progress_threshold:
            self.progress_count = 0  # Reset counter
            best_point = self.best_point[0]
            current_width = self.tr_bounds[1] - self.tr_bounds[0]
            new_width = current_width * 0.5  # Shrink by 50%
            new_lower = np.clip(best_point - new_width, self.bounds[0], self.bounds[1])
            new_upper = np.clip(best_point + new_width, self.bounds[0], self.bounds[1])
            self.tr_bounds = np.vstack((new_lower, new_upper))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]
        self.best_point = (self.X[np.argmin(self.y)], self.y.min())

        while self.n_evals < self.budget:
            # Fit model
            self.gp = self._fit_model(self.X, self.y)

            # Select next point
            next_point = self._select_next_points(1)

            # Evaluate
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

            # Update trust region bounds
            self._update_tr_bounds(next_point[0])

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
