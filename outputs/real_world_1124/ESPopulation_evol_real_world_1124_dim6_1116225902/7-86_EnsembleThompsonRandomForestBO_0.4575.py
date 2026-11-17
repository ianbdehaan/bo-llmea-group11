from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnsembleThompsonRandomForestBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim  # Increased initial design for better ensemble fit
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None  # Surrogate model

    def _sample_points(self, n_points):
        """Generate Latin Hypercube samples scaled to bounds"""
        lh = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = lh.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        """Train a Random Forest with ensemble diversity"""
        model = RandomForestRegressor(
            n_estimators=20,  # Lightweight ensemble
            random_state=int(self.rng.integers(2**31 - 1)),
            verbose=0
        )
        model.fit(X, y)
        return model

    def _select_next_points(self, batch_size):
        """Generate candidates via Latin Hypercube and select via Thompson Sampling"""
        lh = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        candidates = lh.random(n=50)  # Diverse global candidates
        candidates = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidates

        # Estimate predictive distribution via tree outputs
        tree_preds = np.array([tree.predict(candidates) for tree in self.model.estimators_])
        mean = np.mean(tree_preds, axis=0)
        std = np.std(tree_preds, axis=0)

        # Sample from the predictive distribution
        samples = np.random.normal(mean, std, mean.shape)
        best_idx = np.argmin(samples)
        return candidates[best_idx, :][np.newaxis, :]

    def _evaluate_points(self, func, X):
        """Parallelizable function evaluations"""
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        """Merge new evaluations with historical data"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial diverse sampling with Latin Hypercube
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        # Main optimization loop
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

        # Return best found solution
        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
