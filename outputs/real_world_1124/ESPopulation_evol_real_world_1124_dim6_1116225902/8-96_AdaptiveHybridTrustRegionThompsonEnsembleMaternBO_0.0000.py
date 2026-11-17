from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class AdaptiveHybridTrustRegionThompsonEnsembleMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * self.dim + 1  # Expanded initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.best_y = np.inf
        self.region_radius = 2.0  # Initial trust region radius
        self.batch_size = min(4, self.budget // 15)  # More aggressive batch sizing
        self.phase_budget = {
            'exploration': int(0.2 * budget),
            'hybrid': int(0.4 * budget),
            'exploitation': int(0.4 * budget)
        }
        self.current_phase = 'exploration'
        self.last_improvement = 0
        self.refit_interval = 1  # Starts aggressive, adapts later

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            samples = self.sobol_gen.random(n_points)
        else:
            trust_center = np.clip(trust_center, -5.0, 5.0)
            lower = np.clip(trust_center - radius, -5.0, 5.0)
            upper = np.clip(trust_center + radius, -5.0, 5.0)
            samples = self.rng.uniform(low=lower, high=upper, size=(n_points, self.dim))
        return samples

    def _fit_model(self, X, y):
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _calculate_ei(self, candidates):
        mean, std = self.model.predict(candidates, return_std=True)
        std = np.clip(std, 1e-8, None)
        current_best = np.min(self.y)
        z = (mean - current_best) / std
        ei = (mean - current_best) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def _select_next_points(self):
        if self.current_phase == 'exploration':
            # Global EI acquisition
            n_candidates = 300
            candidates = self.sobol_gen.random(n_candidates)
            ei = self._calculate_ei(candidates)
            top_indices = np.argsort(-ei)[:60]
        elif self.current_phase in ['hybrid', 'exploitation']:
            # Trust region Thompson + EI
            n_candidates = 300
            candidates = self._sample_points(n_candidates, self.best_X, self.region_radius)
            # Hybrid scoring: 70% Thompson, 30% EI
            thompson_samples = self.model.sample_y(candidates, n_samples=15)
            thompson_scores = thompson_samples.mean(axis=1)
            ei_scores = self._calculate_ei(candidates)
            scores = 0.7 * thompson_scores + 0.3 * ei_scores
            top_indices = np.argsort(scores)[:60]
        top_candidates = candidates[top_indices]

        # Normalize for clustering
        scaler = StandardScaler()
        scaled_top = scaler.fit_transform(top_candidates)
        kmeans = MiniBatchKMeans(n_clusters=self.batch_size, batch_size=100, random_state=int(self.rng.integers(2**31 - 1)))
        kmeans.fit(scaled_top)
        next_X = scaler.inverse_transform(kmeans.cluster_centers_)
        return next_X

    def _update_phase(self):
        remaining = self.budget - self.n_evals
        if self.current_phase == 'exploration' and remaining <= self.phase_budget['hybrid']:
            self.current_phase = 'hybrid'
        elif self.current_phase == 'hybrid' and remaining <= self.phase_budget['exploitation']:
            self.current_phase = 'exploitation'

    def _update_refit_interval(self):
        if self.last_improvement >= 3:
            # Stalled progress -> refit more frequently
            self.refit_interval = 1
        elif self.last_improvement >= 1:
            # Gradual progress -> medium refit
            self.refit_interval = 2
        else:
            # Rapid progress -> refit less
            self.refit_interval = 4

    def _update_eval_points(self, new_X, new_y):
        self.X = np.vstack([self.X, new_X])
        self.y = np.vstack([self.y, new_y])
        current_best_idx = np.argmin(self.y)
        if self.y[current_best_idx] < self.best_y:
            self.best_y = self.y[current_best_idx]
            self.best_X = self.X[current_best_idx].copy()
            self.last_improvement = 0
            # Expand trust region logarithmically with improvement
            self.region_radius = min(self.region_radius * (1 + np.log(1 + 0.2)), 5.0)
        else:
            self.last_improvement += 1
            self.region_radius = max(self.region_radius / (1 + np.log(1 + 0.2)), 0.5)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = np.zeros((len(X_init), 1))
        for i in range(len(X_init)):
            y_init[i] = func(X_init[i])
            self.n_evals += 1
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Update phase and refit strategy
            self._update_phase()
            self._update_refit_interval()

            # Refit model based on progress-driven schedule
            if self.n_evals % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)

            next_X = self._select_next_points()
            next_y = np.zeros((len(next_X), 1))
            for i in range(len(next_X)):
                next_y[i] = func(next_X[i])
                self.n_evals += 1
            self._update_eval_points(next_X, next_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
