from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DynamicRefitTrustRegionThompsonSamplingBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.refit_interval = 5  # Refit every 5 iterations
        self.last_refit_evals = 0

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=1,  # Reduced from 3 to 1
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _select_next_points(self, batch_size):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=20)  # Reduced from 50 to 20
        candidates = current_lower + candidates * (current_upper - current_lower)

        # Efficient Thompson Sampling with diagonal approximation
        mean, std = self.gp.predict(candidates, return_std=True)
        sample = mean + self.rng.normal(size=mean.shape[0]) * std  # Diagonal approximation
        best_idx = np.argmin(sample)
        return candidates[best_idx, :][np.newaxis, :]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]
        self.best_point = self.X[np.argmin(self.y)]
        self.trust_radius = np.ones(self.dim) * 5.0
        self.last_refit_evals = self.n_evals

        while self.n_evals < self.budget:
            # Refit model at intervals
            if (self.n_evals - self.last_refit_evals) >= self.refit_interval:
                self.gp = self._fit_model(self.X, self.y)
                self.last_refit_evals = self.n_evals

                # Update trust region based on improvement
                current_best = np.min(self.y)
                if hasattr(self, 'previous_best') and self.previous_best > current_best:
                    self.trust_radius *= 1.1
                else:
                    self.trust_radius *= 0.9
                self.trust_radius = np.clip(self.trust_radius, 0.1, 5.0)
                self.previous_best = current_best
                self.best_point = self.X[np.argmin(self.y)]

            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
