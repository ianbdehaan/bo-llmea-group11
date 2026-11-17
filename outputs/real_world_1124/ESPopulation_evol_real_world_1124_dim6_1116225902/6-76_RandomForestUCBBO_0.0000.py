from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math

class RandomForestUCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Latin Hypercube initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.max_iterations = (self.budget - self.n_init) // 2  # Batch size 2
        self.current_iteration = 0

    def _sample_points(self, n_points):
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _sample_sobol_candidates(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single-core for reproducibility
        )
        model.fit(X, y)
        return model

    def _select_next_points(self, batch_size):
        # Generate 100 Sobol candidates for exploration
        candidates = self._sample_sobol_candidates(100)
        n_candidates = candidates.shape[0]
        n_trees = len(self.model.estimators_)

        # Collect predictions from all trees for uncertainty estimation
        predictions = np.zeros((n_trees, n_candidates))
        for i, tree in enumerate(self.model.estimators_):
            predictions[i] = tree.predict(candidates)

        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)

        # Dynamic UCB beta: decays linearly from 2.0 to 0.1
        beta = max(0.1, 2.0 * (1 - self.current_iteration / self.max_iterations))
        ucb = mean_pred + beta * np.sqrt(var_pred)

        # Select top `batch_size` candidates
        best_indices = np.argsort(ucb)[-batch_size:][::-1]
        next_X = candidates[best_indices]
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Latin Hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        while self.n_evals < self.budget:
            batch_size = 2  # Parallel evaluation of 2 points
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
            self.current_iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
