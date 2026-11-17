from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans

class MaximinLHCEIEnsembleBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Maxemin Latin hypercube for initialization
        self.refit_interval = 5  # Model refits every 5 iterations
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(
            d=dim, samples=self.n_init, criterion='maximin', seed=int(self.rng.integers(2**31 - 1))
        )
        self.model = None

    def _sample_points(self, n_points):
        samples = self.lhs_gen.random(n_points) if self.n_evals == 0 else qmc.Sobol(self.dim, scramble=True, seed=int(self.rng.integers(2**31 - 1))).random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # No log-transform for raw objective values to preserve dynamics
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
        candidates = self._sample_points(200)  # 200 Sobol candidates for EI
        mean, std = self.model.predict(candidates, return_std=True)
        best_y = np.min(self.y)

        # Compute Expected Improvement (EI)
        with np.errstate(divide='ignore'):
            Z = (best_y - mean) / std
            ei = (best_y - mean) * norm.cdf(Z) + std * norm.pdf(Z)
        ei = np.ravel(ei)

        # Select top 50 candidates based on EI
        top_indices = np.argsort(ei)[-50:]
        top_candidates = candidates[top_indices]

        # Cluster top candidates to enforce diversity
        kmeans = KMeans(n_clusters=batch_size, random_state=int(self.rng.integers(2**31 - 1)))
        kmeans.fit(top_candidates)
        cluster_indices = []
        for center in kmeans.cluster_centers_:
            dists = np.sum((top_candidates - center)**2, axis=1)
            best_idx = np.argmin(dists)
            cluster_indices.append(best_idx)
        next_X = top_candidates[cluster_indices]
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
        # Initial evaluation with maximin Latin hypercube
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        batch_size = 2  # Evaluate two diverse points in parallel
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)  # Refit periodically

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
