from collections.abc import Callable
from scipy.stats import qmc, norm
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
        self.n_init = 2 * dim + 1  # Improved initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.last_best = np.inf  # Track progress for model refit

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self):
        # Dynamic restarts: 1 when trust region is stable, 3 when expanding
        restarts = 1 if np.allclose(self.trust_radius, self.last_radius) else 2
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=restarts,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(self.X, self.y)
        return gp

    def _select_next_points(self, batch_size=2):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=100)  # More diverse candidates
        candidates = current_lower + candidates * (current_upper - current_lower)

        mean, std = self.gp.predict(candidates, return_std=True)
        mean = mean.flatten()
        std = std.flatten()

        current_best = np.min(self.y)
        mask = std == 0
        z = (current_best - mean) / std
        z = np.clip(z, -8, 8)
        ei = np.zeros_like(mean)
        ei[~mask] = (current_best - mean[~mask]) * norm.cdf(z[~mask]) + std[~mask] * norm.pdf(z[~mask])
        ei[mask] = 0.0

        top_indices = np.argsort(ei)[::-1][:batch_size]
        return candidates[top_indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]
        self.best_point = self.X[np.argmin(self.y)]
        self.trust_radius = np.ones(self.dim) * 1.0  # Smaller initial radius
        self.last_radius = self.trust_radius.copy()

        while self.n_evals < self.budget:
            self.gp = self._fit_model()
            next_points = self._select_next_points(2)  # Batch of 2
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            self.n_evals += 2

            # Update trust region based on progress
            new_best = np.min(self.y)
            improved = new_best < self.last_best
            self.last_best = new_best

            if improved:
                self.trust_radius *= 1.05  # Smoother expansion
            else:
                self.trust_radius *= 0.95  # Smoother contraction
            self.trust_radius = np.clip(self.trust_radius, 0.2, 5.0)  # Tighter lower bound
            self.best_point = self.X[np.argmin(self.y)]
            self.last_radius = self.trust_radius.copy()

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
