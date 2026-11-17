from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import KMeans

class DynamicTrustRegionLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.eta = 1.0  # Initial exploration parameter for LCB

    def _sample_points(self, n_points):
        lhd = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = lhd.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * RBF(length_scale=np.ones(self.dim), 
                           length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _select_next_points(self, batch_size=3, n_candidates=100):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        lhd = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        candidates = lhd.random(n=n_candidates)
        candidates = current_lower + candidates * (current_upper - current_lower)

        # Compute mean and std for each candidate
        mean, std = self.gp.predict(candidates, return_std=True)
        mean = mean.flatten()
        std = std.flatten()

        # Lower Confidence Bound (LCB) with dynamic exploration weight
        lcb = mean - self.eta * std
        best_idx = np.argsort(lcb)[-batch_size*3:]  # Top 3*batch_size candidates

        # Cluster top candidates and select diverse batch
        top_candidates = candidates[best_idx]
        kmeans = KMeans(n_clusters=batch_size, n_init=1, random_state=int(self.rng.integers(2**31 - 1)))
        kmeans.fit(top_candidates)
        cluster_centers = kmeans.cluster_centers_
        selected_idx = np.argmin(np.sum((top_candidates[:, np.newaxis] - cluster_centers) ** 2, axis=2), axis=0)
        return top_candidates[selected_idx]

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
            self.gp = self._fit_model(self.X, self.y)
            next_points = self._select_next_points(3)  # Select 3 diverse points
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            self.n_evals += 3

            # Update trust region based on batch progress
            batch_improvement = np.min(new_y) < np.min(self.y)
            if batch_improvement:
                self.trust_radius *= 1.1
                self.eta *= 0.9  # Decrease exploration if improvement is made
            else:
                self.trust_radius *= 0.9
                self.eta *= 1.1  # Increase exploration if no progress
            self.trust_radius = np.clip(self.trust_radius, 0.1, 5.0)
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
