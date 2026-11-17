from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import math

class DynamicLCLowestRandomForestBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = self.dim + 2  # Latin Hypercube initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_y = float('inf')
        self.refit_required = True  # Only refit when a new best is found

    def _sample_points(self, n_points):
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        """Train Random Forest with bootstrap sampling for uncertainty estimation."""
        n_estimators = 20  # Reduced for efficiency
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            bootstrap=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _predict_with_std(self, X):
        """Estimate predictive mean and standard deviation using tree variance."""
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(tree_predictions, axis=0)
        std = np.std(tree_predictions, axis=0)
        return mean, std

    def _select_next_points(self, batch_size):
        """Generate diverse candidates using dynamic LCB and clustering."""
        n_candidates = 200  # Generate diverse candidates
        candidates = self._sample_points(n_candidates)
        mean, std = self._predict_with_std(candidates)

        # Dynamic beta: decreases as iterations progress
        beta = 1.0 - (self.n_evals / self.budget)
        lcb = mean - beta * std

        # Select top candidates and cluster them
        top_indices = np.argsort(lcb)[:50]  # Take top 50 candidates
        top_candidates = candidates[top_indices]

        # Reduce dimensionality to improve clustering performance
        if top_candidates.shape[1] > 2:
            pca = PCA(n_components=2, random_state=int(self.rng.integers(2**31 - 1)))
            reduced = pca.fit_transform(top_candidates)
        else:
            reduced = top_candidates

        # Cluster using MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=batch_size,
            random_state=int(self.rng.integers(2**31 - 1)),
            batch_size=min(len(reduced), 100)
        )
        kmeans.fit(reduced)
        cluster_centers = kmeans.cluster_centers_

        # Map cluster centers back to original space (if PCA used)
        if top_candidates.shape[1] > 2:
            cluster_centers = pca.inverse_transform(cluster_centers)

        # Ensure cluster centers are within bounds
        cluster_centers = np.clip(cluster_centers, self.bounds[0], self.bounds[1])
        return cluster_centers

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
            if y[i] < self.best_y:
                self.best_y = y[i]
                self.refit_required = True  # Trigger model refit on improvement
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_y = np.min(y_init)
        self.model = self._fit_model(self.X, self.y)

        # Main loop
        batch_size = 2  # Parallel evaluation of 2 points per iteration
        while self.n_evals < self.budget:
            # Refit only when a new best is found
            if self.refit_required:
                self.model = self._fit_model(self.X, self.y)
                self.refit_required = False  # Reset flag

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
