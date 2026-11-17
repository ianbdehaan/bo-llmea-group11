import numpy as np
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from collections.abc import Callable

class LatinForestThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 2 * dim)  # Ensure sufficient initial samples
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))  # Seeded RNG

    def _sample_points(self, n_points):
        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return scaled_samples

    def _fit_model(self, X, y):
        # Random Forest with moderate complexity and random state
        rf = RandomForestRegressor(
            n_estimators=10,
            max_features='sqrt',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        rf.fit(X, y.ravel())
        return rf

    def _acquisition_function(self, X):
        # Sample a random tree from the forest for Thompson Sampling
        tree_idx = self.rng.integers(len(self.model.estimators_))
        sampled_tree = self.model.estimators_[tree_idx]
        predictions = sampled_tree.predict(X)
        return predictions.reshape(-1, 1)  # For minimization

    def _select_next_points(self, batch_size):
        # Generate diverse candidates and select top batch_size
        n_candidates = 100  # More candidates for batch selection
        candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acq_values, axis=0)[:batch_size].ravel()
        best_candidates = candidates[best_indices]
        return best_candidates

    def _evaluate_points(self, func, X):
        # Evaluate func for each row in X
        y = np.array([func(x) for x in X])
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with Latin Hypercube samples
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)
        self.n_evals = len(self.X)
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        batch_size = 10  # Match parallel evaluation constraint
        while self.n_evals + batch_size <= self.budget:
            self.model = self._fit_model(self.X, self.y)
            next_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            current_y = new_y[:, 0]
            for i in range(batch_size):
                if current_y[i] < best_y:
                    best_y = current_y[i]
                    best_x = next_points[i]

        # Handle remaining evaluations if budget not divisible by batch_size
        remaining = self.budget - self.n_evals
        if remaining > 0:
            next_points = self._select_next_points(remaining)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            current_y = new_y[:, 0]
            for i in range(remaining):
                if current_y[i] < best_y:
                    best_y = current_y[i]
                    best_x = next_points[i]

        return best_y, best_x
