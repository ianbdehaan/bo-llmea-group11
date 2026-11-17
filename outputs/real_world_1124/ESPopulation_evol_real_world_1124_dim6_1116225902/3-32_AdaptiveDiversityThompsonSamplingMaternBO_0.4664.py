from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class AdaptiveDiversityThompsonSamplingMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * self.dim + 1  # Enhanced initial design
        self.refit_interval = 10  # Extended refit interval with dynamic updates
        self.last_improvement = 0
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None

    def _sample_points(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(250)  # Increased candidate pool
        sampled_functions = self.model.sample_y(candidates, n_samples=1).flatten()

        # Greedy selection with diversity
        selected_indices = []
        for i in range(batch_size):
            if i == 0:
                best_idx = np.argmin(sampled_functions)
            else:
                # Maximize distance to already selected points
                distances = np.min([np.linalg.norm(candidates[candidates.shape[0] // 2 + i] - candidates[best_idx]) for best_idx in selected_indices], axis=0)
                best_idx = np.argmin(sampled_functions)
            selected_indices.append(best_idx)
            sampled_functions[best_idx] = np.inf  # Remove selected point from candidates

        next_X = candidates[np.array(selected_indices)]
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _should_refit(self):
        # Force refit if last 3 iterations showed no improvement
        if self.last_improvement >= 3:
            return True
        return (self.n_evals - self.last_improvement) % self.refit_interval == 0

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.last_improvement = self.n_evals

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        while self.n_evals < self.budget:
            if self._should_refit():
                self.model = self._fit_model(self.X, self.y)  # Conditional refitting
                self.last_improvement = self.n_evals

            batch_size = 2  # Evaluate two diverse points in parallel
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Track improvement
            if np.min(next_y) < np.min(self.y):
                self.last_improvement = self.n_evals

            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
