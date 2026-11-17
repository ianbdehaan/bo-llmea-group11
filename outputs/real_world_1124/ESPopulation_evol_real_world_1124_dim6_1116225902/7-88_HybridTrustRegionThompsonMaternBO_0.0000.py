from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class HybridTrustRegionThompsonMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Reduced initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.best_y = np.inf
        self.region_radius = 2.0  # Initial trust region radius
        self.batch_size = max(2, min(4, int(math.sqrt(budget) / 5)))  # Adaptive batch size

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            samples = self.sobol_gen.random(n_points)
        else:
            trust_center = np.clip(trust_center, -5.0, 5.0)
            samples = self.rng.normal(loc=trust_center, scale=radius, size=(n_points, self.dim))
            samples = np.clip(samples, -5.0, 5.0)
        return samples

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
        candidates = self._sample_points(200, self.best_X, self.region_radius)
        candidates = self.bounds[0] + candidates * (self.bounds[1] - self.bounds[0])

        # Alternate between Thompson Sampling and Expected Improvement
        if self.n_evals % 2 == 0:
            # Thompson Sampling with 15 ensemble samples
            sampled_funcs = self.model.sample_y(candidates, n_samples=15)
            idxs = np.argmin(sampled_funcs, axis=1)
            thompson_candidates = candidates[idxs]
        else:
            # Expected Improvement acquisition function
            mu, sigma = self.model.predict(candidates, return_std=True)
            mu = np.exp(mu)  # Reverse log transformation
            best_mu = self.best_y
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = best_mu - mu
                Z = imp / (sigma + 1e-8)
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            valid_mask = sigma > 1e-8
            ei[~valid_mask] = 0.0
            ei = -ei  # Minimization
            top_indices = np.argsort(ei)[:, None]
            ei_candidates = candidates[top_indices]

        # Greedy diversity selection for batch
        if self.n_evals % 2 == 0:
            candidates_pool = thompson_candidates
        else:
            candidates_pool = ei_candidates
        batch = []
        while len(batch) < self.batch_size and len(candidates_pool) > 0:
            if not batch:
                farthest_idx = np.argmax(np.sum((candidates_pool - np.mean(candidates_pool, axis=0))**2, axis=1))
                selected = candidates_pool[farthest_idx]
            else:
                distances = np.min(np.linalg.norm(candidates_pool[:, None, :] - np.array(batch)[None, :, :], axis=2), axis=1)
                farthest_idx = np.argmax(distances)
                selected = candidates_pool[farthest_idx]
            batch.append(selected)
            candidates_pool = np.delete(candidates_pool, farthest_idx, axis=0)
        return np.array(batch)

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
            self.best_y = np.min(new_y)
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            current_min = np.min(self.y)
            if current_min < self.best_y:
                self.best_X = self.X[np.argmin(self.y)].copy()
                self.best_y = current_min
                # Expand trust region based on improvement and model uncertainty
                mu, sigma = self.model.predict(self.X, return_std=True)
                mu = np.exp(mu)  # Reverse log transformation
                improvement = self.best_y - mu
                sigma_threshold = np.median(sigma)  # Use median uncertainty as threshold
                if np.any(improvement > sigma_threshold):
                    self.region_radius = min(self.region_radius * 1.2, 5.0)
            else:
                # Contract trust region if no improvement and uncertainty is low
                mu, sigma = self.model.predict(self.X, return_std=True)
                mu = np.exp(mu)
                if np.all(sigma < 0.1):
                    self.region_radius = max(self.region_radius * 0.8, 0.5)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with trust region adaptation
        iteration = 0
        while self.n_evals < self.budget:
            # Adaptive refitting based on trust region size
            if self.region_radius > 1.0 or iteration % 2 == 0:
                self.model = self._fit_model(self.X, self.y)
            
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
