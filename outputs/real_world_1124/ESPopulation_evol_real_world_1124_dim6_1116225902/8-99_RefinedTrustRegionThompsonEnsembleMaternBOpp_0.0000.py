from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans
import math

class RefinedTrustRegionThompsonEnsembleMaternBOpp:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * self.dim + 1  # Maintains efficient initial design
        self.refit_interval = 3  # Base refit frequency, may be adjusted
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.region_radius = 2.0
        self.batch_size = min(4, self.budget // 10)
        
        # Adaptive candidate count based on dimensionality
        self.candidate_per_dim = 50 + (4 - dim) * 10  # 120 for dim=6

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            return self.sobol_gen.random(n_points)
        
        trust_center = np.clip(trust_center, -5.0, 5.0)
        base_center = np.tile(trust_center, (n_points, 1))
        
        # Hybrid sampling: 75% uniform in hypercube, 25% normal
        n_uniform = int(0.75 * n_points)
        n_normal = n_points - n_uniform
        
        uniform_samples = base_center + radius * (self.rng.rand(n_uniform, self.dim) - 0.5)
        normal_samples = self.rng.normal(loc=trust_center, scale=radius/2, size=(n_normal, self.dim))
        samples = np.vstack([uniform_samples, normal_samples])
        return np.clip(samples, -5.0, 5.0)

    def _fit_model(self, X, y):
        y_log = np.log(y + 1e-8)
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_log)
        return model

    def _select_next_points(self):
        n_candidates = max(100, self.candidate_per_dim * self.dim)
        candidates = self._sample_points(n_candidates, self.best_X, self.region_radius)
        
        # Thompson Sampling with 15 ensemble samples
        sampled_funcs = self.model.sample_y(candidates, n_samples=15)
        idxs = np.argmin(sampled_funcs, axis=1)
        
        # Select top 30% candidates
        top_indices = np.argsort(sampled_funcs.mean(axis=1))[:int(0.3 * len(candidates))]
        top_candidates = candidates[top_indices]
        
        # MiniBatchKMeans with 80% subsampling for efficiency
        if len(top_candidates) >= self.batch_size:
            kmeans = MiniBatchKMeans(
                n_clusters=self.batch_size, 
                batch_size=int(0.8 * len(top_candidates)), 
                random_state=int(self.rng.integers(2**31 - 1))
            )
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
            
            # Update trust region based on progress
            if np.min(new_y) < np.min(self.y):
                self.best_X = new_X[np.argmin(new_y)].copy()
                self.region_radius = min(self.region_radius * 1.25, 5.0)  # 25% expansion
            else:
                self.region_radius = max(self.region_radius * 0.75, 0.2)  # 25% contraction

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.model = self._fit_model(self.X, self.y)
        iteration = 0

        while self.n_evals < self.budget:
            # Dynamic refitting: every 2 iterations when trust region is large (>1.0) or progress is made
            if (iteration % 2 == 0 and self.region_radius > 1.0) or (iteration % self.refit_interval == 0):
                self.model = self._fit_model(self.X, self.y)
            
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            iteration += 1

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
