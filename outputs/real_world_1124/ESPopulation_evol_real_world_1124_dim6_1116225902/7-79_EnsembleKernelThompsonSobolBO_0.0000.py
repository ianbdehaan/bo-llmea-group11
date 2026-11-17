from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import sklearn.metrics
import math

class EnsembleKernelThompsonSobolBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim  # LHS for 6D
        self.refit_interval = 10  # Refit every 10 iterations if no progress
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.trust_radius = 0.5  # Initial trust radius as 10% of the domain width (5*0.1=0.5)
        self._progress_threshold = 1e-4  # Minimal improvement to trigger refit
        self.model_ensemble = []
        self.trust_center = None

    def _sample_points(self, n_points, trust_center=None, trust_radius=None):
        if trust_center is None:
            samples = self.lhs_gen.random(n_points)
        else:
            # Generate within trust region (hypercube)
            domain_width = self.bounds[1] - self.bounds[0]
            half_radius = trust_radius * domain_width / 2.0
            samples = self.rng.uniform(-half_radius, half_radius, (n_points, self.dim))
            samples += trust_center
            # Clip to bounds
            samples = np.clip(samples, self.bounds[0], self.bounds[1])
        scaled_samples = samples  # LHS already in [0,1]
        return scaled_samples

    def _fit_model(self, X, y):
        y_log = np.log(y + 1e-8)  # Log-transform for numerical stability

        # Ensemble of two kernels: Matérn-5/2 and RBF with ARD
        kernel1 = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        kernel2 = RBF(length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model1 = GaussianProcessRegressor(
            kernel=kernel1,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model2 = GaussianProcessRegressor(
            kernel=kernel2,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model1.fit(X, y_log)
        model2.fit(X, y_log)
        return [model1, model2]

    def _select_next_points(self, batch_size):
        if self.trust_center is None:
            trust_center = self.X[np.argmin(self.y)].reshape(1, -1)  # Initialize with best point
        else:
            trust_center = self.trust_center

        candidates = self._sample_points(200, trust_center=trust_center, trust_radius=self.trust_radius)
        
        # Thompson Sampling from each model in the ensemble
        model1, model2 = self.model_ensemble
        sampled1 = model1.sample_y(candidates, n_samples=1).flatten()
        sampled2 = model2.sample_y(candidates, n_samples=1).flatten()

        # Combine samples and select top candidates
        all_candidates = np.vstack([candidates, candidates])
        all_samples = np.hstack([sampled1, sampled2])
        top_indices = np.argsort(all_samples)[:batch_size]

        # Cluster selected candidates to ensure diversity
        top_candidates = all_candidates[top_indices]
        _, cluster_idx = sklearn.cluster.vq(top_candidates, sklearn.cluster.kmeans(top_candidates, 1, random_state=int(self.rng.integers(2**31 - 1)))[0])
        unique_clusters = np.unique(cluster_idx)
        next_X = top_candidates[unique_clusters[:batch_size]]
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

    def _should_refit(self):
        # Trigger refit if progress is minimal or trust radius is large
        if self.n_evals >= self.budget:
            return False
        recent_best = self.y[np.argmin(self.y)]
        if self.X.shape[0] >= 2:
            prev_best = np.partition(self.y, 1, axis=0)[1, 0]  # 2nd best
            if abs(prev_best - recent_best) < self._progress_threshold:
                return True
        return (self.n_evals % self.refit_interval) == 0

    def _update_trust_region(self, new_X, new_y):
        # Update trust region center to best point
        best_idx = np.argmin(np.vstack([self.y, new_y]), axis=0)
        if best_idx == 0:  # No new improvement
            self.trust_radius *= 0.95  # Shrink region if no progress
        else:
            new_center = new_X[best_idx - 1].reshape(1, -1)  # -1 because best_idx is 1-based
            self.trust_center = new_center
            # Expand if within trust region
            dist = np.max(np.abs(new_center - self.X[np.argmin(self.y)]))
            if dist < self.trust_radius * 0.1:
                self.trust_radius = min(self.trust_radius * 1.1, 0.5)  # Max 50% of domain width

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with LHS
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Initialize trust region and model
        self.trust_center = self.X[np.argmin(self.y)].reshape(1, -1)
        self.model_ensemble = self._fit_model(self.X, self.y)

        iteration = 0
        while self.n_evals < self.budget:
            if self._should_refit():
                self.model_ensemble = self._fit_model(self.X, self.y)

            batch_size = 2  # Batch size adapts to trust region size
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update trust region based on new evaluations
            self._update_trust_region(next_X, next_y)

            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
