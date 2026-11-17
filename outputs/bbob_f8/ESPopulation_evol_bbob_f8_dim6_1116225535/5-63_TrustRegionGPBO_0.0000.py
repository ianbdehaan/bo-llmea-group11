from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

class TrustRegionGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim  # Larger initialization for better GP fitting
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(nu=1.5, length_scale_bounds="fixed") + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            n_restarts_optimizer=1,  # Fast model fitting
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Sample from GP posterior for Thompson Sampling
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        return mu + sigma * self.rng.normal(size=mu.shape)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(30, bounds=self.trust_bounds)  # Reduced candidates
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)[:, None]
        top_candidates = candidates[idxs[-3:]]  # Top 3 candidates

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(self.trust_bounds[0], self.trust_bounds[1])]
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)
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
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with Latin Hypercube design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize trust region
        trust_radius = 1.0  # Initial radius
        self.trust_bounds = np.array([
            np.clip(best_x - trust_radius, -5.0, 5.0),
            np.clip(best_x + trust_radius, -5.0, 5.0)
        ])

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
                # Expand trust region
                new_radius = trust_radius * 1.3  # Moderate expansion
                new_radius = min(new_radius, 5.0 - best_x[0])
                self.trust_bounds = np.array([
                    np.clip(best_x - new_radius, -5.0, 5.0),
                    np.clip(best_x + new_radius, -5.0, 5.0)
                ])
                trust_radius = new_radius
            else:
                # Shrink trust region
                new_radius = trust_radius * 0.8  # Moderate contraction
                self.trust_bounds = np.array([
                    np.clip(best_x - new_radius, -5.0, 5.0),
                    np.clip(best_x + new_radius, -5.0, 5.0)
                ])
                trust_radius = new_radius

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
