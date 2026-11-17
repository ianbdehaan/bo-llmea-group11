from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class HybridThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim  # Adaptive initialization for better coverage
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.best_x = None

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = RBF(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Reduced restarts for efficiency
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)  # Avoid division by zero
        samples = mu + sigma * self.rng.normal(size=(3, mu.shape[0])).T  # 3 samples
        return samples  # Shape (n_points, 3)

    def _select_next_points(self, batch_size):
        global_candidates = self._sample_points(50)  # Global Sobol candidates
        local_candidates = np.zeros((50, self.dim))
        for i in range(50):
            noise = self.rng.normal(0, 0.3, self.dim)
            local_candidate = self.best_x + noise
            local_candidate = np.clip(local_candidate, self.bounds[0], self.bounds[1])
            local_candidates[i] = local_candidate
        candidates = np.vstack([global_candidates, local_candidates])  # 100 candidates

        acq_samples = self._acquisition_function(candidates)  # Shape (100, 3)
        optimized_points = []

        for sample_idx in range(acq_samples.shape[1]):  # For each sample
            sample_acq = acq_samples[:, sample_idx]
            idxs = np.argsort(sample_acq, axis=0)
            top_candidates = candidates[idxs[-5:]]  # Top 5 per sample

            for x0 in top_candidates:
                def acq_wrapper(x):
                    return -self._acquisition_function(x.reshape(1, -1))[:, sample_idx].item()
                res = minimize(
                    acq_wrapper, 
                    x0, 
                    bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                    method='L-BFGS-B'
                )
                if res.success:
                    optimized_points.append(res.x)

        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)  # Shape (n_points, 3)
        best_idx = np.argmax(acq_values_opt.mean(axis=1))  # Average over samples
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        y = np.zeros((n_new, 1))
        for i in range(n_new):
            y[i, 0] = func(X[i])
        self.n_evals += n_new
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
            self.best_x = new_X[np.argmin(new_y)].copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            if np.min(new_y) < np.min(self.y):
                self.best_x = new_X[np.argmin(new_y)].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.model = self._fit_model(self.X, self.y)

        return self.y.min(), self.best_x
