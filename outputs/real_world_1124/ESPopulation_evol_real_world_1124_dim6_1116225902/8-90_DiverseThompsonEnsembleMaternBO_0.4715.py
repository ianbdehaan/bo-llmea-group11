from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans
import math

class DiverseThompsonEnsembleMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * self.dim + 1  # Enhanced initial design
        self.refit_interval = 5  # Default refit interval
        self.threshold_progress = 1e-3  # Progress threshold for dynamic refitting
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None

    def _sample_points(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        y_log = np.log(y + 1e-8)  # Log-transform for numerical stability
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_log)
        return model

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(200)  # 200 Sobol candidates
        # Generate 15 diverse Thompson samples
        sampled_functions = self.model.sample_y(candidates, n_samples=15)
        # Collect best candidate per sample
        best_indices = np.argmin(sampled_functions, axis=0)
        candidate_pool = candidates[best_indices]
        
        # Cluster to select diverse batch
        n_clusters = min(batch_size, len(candidate_pool))
        if n_clusters >= 2:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=50, n_init=1)
            kmeans.fit(candidate_pool)
            # Select cluster centers as diverse candidates
            next_X = kmeans.cluster_centers_
        else:
            next_X = candidate_pool[:n_clusters]
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

    def _should_refit(self, best_y):
        # Check if recent progress justifies refitting
        if self.n_evals < 5:
            return True
        prev_best = self.y[np.argmin(self.y)]
        return (best_y < prev_best - self.threshold_progress) or (self.n_evals % self.refit_interval == 0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        while self.n_evals < self.budget:
            best_y = np.min(self.y)
            if self._should_refit(best_y):
                self.model = self._fit_model(self.X, self.y)  # Refit dynamically
            
            batch_size = 3  # Evaluate 3 points in parallel
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
