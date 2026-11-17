from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class LocalGPMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Latin Hypercube initial design
        self.refit_interval = 1  # Refit local model every iteration
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lh_design = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.region_radius = 2.0  # Initial axis-aligned hypercubic trust region radius
        self.batch_size = min(3, self.budget // 20)  # Small adaptive batch size

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            samples = self.lh_design.random(n_points)
        else:
            # Generate Sobol samples within axis-aligned hypercube
            lower = np.clip(trust_center - radius, -5.0, 5.0)
            upper = np.clip(trust_center + radius, -5.0, 5.0)
            samples = self.rng.uniform(low=lower, high=upper, size=(n_points, self.dim))
        return samples

    def _fit_model(self, X, y):
        # Filter local data within current trust region
        mask = np.all((X >= (self.best_X - self.region_radius)) & (X <= (self.best_X + self.region_radius)), axis=1)
        local_X, local_y = X[mask], y[mask]
        if len(local_X) < self.dim + 1:  # Fallback to global if insufficient local data
            local_X, local_y = X, y

        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,  # Faster fit for local model
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(local_X, local_y)
        return model

    def _expected_improvement(self, X, model, best_y):
        mu, sigma = model.predict(X, return_std=True)
        with np.errstate(divide='warn'):
            imp = mu - best_y
            Z = imp / (sigma + 1e-8)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.] = 0.
        return ei

    def _select_next_points(self):
        candidates = self._sample_points(200, self.best_X, self.region_radius)
        ei = self._expected_improvement(candidates, self.model, np.min(self.y))
        # Select top 30% candidates
        top_indices = np.argsort(ei)[-int(0.3 * len(candidates)):]
        top_candidates = candidates[top_indices]

        # Cluster to get diverse batch
        if len(top_candidates) >= self.batch_size:
            kmeans = KMeans(n_clusters=self.batch_size, n_init=1, random_state=int(self.rng.integers(2**31 - 1)))
            kmeans.fit(top_candidates)
            next_X = kmeans.cluster_centers_
        else:
            next_X = top_candidates[:self.batch_size]
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
            self.best_X = new_X[np.argmin(new_y)].copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            if np.min(new_y) < np.min(self.y):
                self.best_X = new_X[np.argmin(new_y)].copy()
                self.region_radius = min(self.region_radius * 1.1, 5.0)  # Expand
            else:
                self.region_radius = max(self.region_radius * 0.9, 0.5)  # Contract

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with trust region adaptation
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)  # Refit local model

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
