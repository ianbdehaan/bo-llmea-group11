from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnsembleLCBAdaptiveBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)  # Ensure at least 5 initial points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.beta = 2.0  # Initial beta for Lower Confidence Bound
        self.min_beta = 0.5
        self.max_beta = 5.0
        self.no_improvement_count = 0  # Track progress for adaptive beta

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Latin Hypercube candidates within specified bounds."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Train a Random Forest ensemble for mean and variance prediction."""
        model = RandomForestRegressor(
            n_estimators=20,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1  # Enable parallel training
        )
        model.fit(X, y.ravel())
        return model

    def _predict(self, X):
        """Estimate mean and variance using tree-level predictions."""
        preds = np.array([tree.predict(X).ravel() for tree in self.model.estimators_])
        mean = np.mean(preds, axis=0)
        var = np.var(preds, axis=0)
        return mean, var

    def _acquisition_function(self, X):
        """Lower Confidence Bound for minimization."""
        mean, var = self._predict(X)
        std = np.sqrt(var)
        lcb = mean - self.beta * std
        return lcb

    def _select_next_points(self):
        """Generate 100 global candidates and select the top LCB."""
        candidates = self._sample_points(100)  # 100 LHS candidates
        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmin(acq_vals)  # Minimize LCB
        next_X = candidates[best_idx].reshape(1, -1)
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points, respecting evaluation budget."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update stored evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation with Latin Hypercube Sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Initialize best solution and model
        best_idx = np.argmin(self.y)
        self.model = self._fit_model(self.X, self.y)
        previous_best = np.min(self.y).copy()

        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Track progress for adaptive beta
            current_best = np.min(self.y)
            if current_best < previous_best:
                self.no_improvement_count = 0
                previous_best = current_best
            else:
                self.no_improvement_count += 1

            # Dynamically adjust beta
            if self.no_improvement_count >= 3:
                self.beta = min(self.beta + 0.5, self.max_beta)  # Increase exploration
            else:
                self.beta = max(self.beta - 0.1, self.min_beta)  # Decrease exploration

            # Refit model
            self.model = self._fit_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
