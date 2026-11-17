from collections.abc import Callable
from scipy.stats import qmc, norm
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
        self.n_init = 2 * dim  # Initial design size
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.eval_counter = 0  # Tracks iterations since last model fit

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
            n_restarts_optimizer=1 if self.eval_counter >= 2 else 3,  # Fewer restarts when not refitting
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, np.log(y + 1e-8))  # Log-transform target for stability
        return gp

    def _select_next_points(self, batch_size):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=200)  # More candidates for better EI optimization
        candidates = current_lower + candidates * (current_upper - current_lower)

        # Compute mean and std for each candidate
        mean, std = self.gp.predict(candidates, return_std=True)
        mean = mean.flatten()
        std = std.flatten()

        # Expected Improvement calculation with log-transform
        current_best = np.min(np.exp(self.y))
        mask = std == 0
        z = (np.log(current_best) - mean) / std  # EI on log scale
        z = np.clip(z, -8, 8)  # Avoid numerical overflow
        ei = np.zeros_like(mean)
        ei[~mask] = (np.log(current_best) - mean[~mask]) * norm.cdf(z[~mask]) + std[~mask] * norm.pdf(z[~mask])
        ei[mask] = 0.0

        # Select batch_size best candidates using greedy approach
        best_indices = np.argsort(ei)[-batch_size:]
        return candidates[best_indices]

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
        self.trust_radius = np.ones(self.dim) * 1.0  # Smaller initial trust radius

        while self.n_evals < self.budget:
            # Determine batch size based on remaining budget
            batch_size = 2 if (self.budget - self.n_evals) > 10 else 1

            # Fit model only when progress stalls or needed for new batch
            if self.eval_counter >= 2 or self.gp is None:
                self.gp = self._fit_model(self.X, self.y)
                self.eval_counter = 0
            else:
                self.eval_counter += 1

            # Select and evaluate next points
            next_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            self.n_evals += batch_size

            # Update trust region based on progress
            prev_best = np.min(self.y)
            new_best = np.min(self.y)
            if new_best < prev_best:
                self.trust_radius *= 1.1  # Expand if improving
            else:
                self.trust_radius *= 0.9  # Contract if no improvement
            self.trust_radius = np.clip(self.trust_radius, 0.1, 5.0)
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
