from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans
import math

class AdaptiveEnsembleTrustRegionThompsonMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * self.dim + 1  # Improved initial design for 6D
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.region_radius = 2.5  # Slightly larger initial radius
        self.base_batch_size = min(3, self.budget // 15)  # Conservative base batch size

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            # Global sampling using Sobol
            samples = self.sobol_gen.random(n_points)
        else:
            # Local sampling using Latin Hypercube within trust region
            trust_center = np.clip(trust_center, -5.0, 5.0)
            lower = np.clip(trust_center - radius, -5.0, 5.0)
            upper = np.clip(trust_center + radius, -5.0, 5.0)
            samples = self.lhs_gen.random(n=n_points)  # Uniform distribution
            samples = lower + (upper - lower) * samples
        return samples

    def _fit_model(self, X, y, is_local=False):
        # Use log-transformed y for numerical stability
        y_log = np.log(y + 1e-8)
        
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,  # Increased for better hyperparameter optimization
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_log)
        return model

    def _select_next_points(self):
        # Generate candidates with adaptive ensemble size
        progress = np.any(np.min(self.y) < self.y[-self.batch_size:])
        ensemble_size = 25 if progress else 15  # Larger ensembles during progress
        candidates = self._sample_points(250, self.best_X, self.region_radius)
        
        # Thompson Sampling with adaptive ensemble
        sampled_funcs = self.model.sample_y(candidates, n_samples=ensemble_size)
        idxs = np.argmin(sampled_funcs, axis=1)
        
        # Select top 40% candidates for diversity
        top_indices = np.argsort(sampled_funcs.mean(axis=1))[:100]
        top_candidates = candidates[top_indices]
        
        # Dynamic cluster count based on recent progress
        cluster_count = self.base_batch_size + (0 if progress else 1)  # Add 1 cluster when stalled
        kmeans = MiniBatchKMeans(n_clusters=cluster_count, n_init=1, batch_size=100, random_state=int(self.rng.integers(2**31 - 1)))
        kmeans.fit(top_candidates)
        next_X = kmeans.cluster_centers_
        
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
                # Aggressive expansion during progress
                self.region_radius = min(self.region_radius * 1.25, 5.0)
            else:
                # Conservative contraction when stalled
                self.region_radius = max(self.region_radius * 0.85, 0.5)

        # Double refitting strategy
        if self.n_evals % 2 == 0:  # Even-numbered iterations: local refit
            self.model = self._fit_model(self.X, self.y, is_local=True)
        else:  # Odd-numbered iterations: full refit
            self.model = self._fit_model(self.X, self.y)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with improved design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Main loop with adaptive trust region and ensemble
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
