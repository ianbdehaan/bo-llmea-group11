from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class TrustRegionSobolRFEnsembleEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.ensemble = None  # Random forest ensemble
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.iteration = 0

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_ensemble(self, X, y):
        """Create an ensemble of 10 random forests with varying hyperparameters"""
        models = []
        for i in range(10):
            max_depth = 5 + i  # Varying depth
            min_samples_split = max(2, 50 // (i+1))  # Varying leaf size
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            model.fit(X, y)
            models.append(model)
        return models

    def _select_next_points(self, batch_size):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = current_lower + sobol.random(n=100) * (current_upper - current_lower)

        # Get mean prediction from the ensemble
        ensemble_preds = np.array([model.predict(candidates) for model in self.ensemble])
        mean_pred = np.mean(ensemble_preds, axis=0)
        std_pred = np.std(ensemble_preds, axis=0)
        
        # Expected Improvement calculation
        current_best = np.min(self.y)
        mask = std_pred == 0
        z = (current_best - mean_pred) / std_pred
        z = np.clip(z, -8, 8)  # Avoid numerical overflow
        ei = np.zeros_like(mean_pred)
        ei[~mask] = (current_best - mean_pred[~mask]) * norm.cdf(z[~mask]) + std_pred[~mask] * norm.pdf(z[~mask])
        ei[mask] = 0.0  # No improvement if variance is zero
        best_idx = np.argmax(ei)
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

        while self.n_evals < self.budget:
            # Refit ensemble less frequently to save computation
            if self.iteration % 3 == 0 or self.ensemble is None:
                self.ensemble = self._fit_ensemble(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1
            self.iteration += 1

            # Update trust region based on progress
            prev_best = np.min(self.y)
            new_best = np.min(self.y)
            if new_best < prev_best:
                self.trust_radius *= 1.1
            else:
                self.trust_radius *= 0.9
            self.trust_radius = np.clip(self.trust_radius, 0.1, 5.0)
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
