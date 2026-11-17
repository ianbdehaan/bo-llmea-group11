from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

class HybridGlobalLocalGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim  # Maximin Sobol initialization for valley coverage
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.global_model = None
        self.local_model = None
        self.best_x = None
        self.best_y = np.inf

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_global_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=5/2)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=1e-8,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _fit_local_model(self, X, y):
        # Select points near best_x within 1.5*std of current model
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        distances = np.linalg.norm(X - best_point, axis=1)
        _, std = self.global_model.predict(X, return_std=True)
        valid_mask = (distances < 1.5 * np.mean(std)) & (y < self.best_y + 2.0)
        X_local = X[valid_mask]
        
        if len(X_local) < self.dim + 1:
            return None  # Skip local model if insufficient data

        kernel = RBF(length_scale=0.1 * np.ones(self.dim))  # Smaller length scale
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            alpha=1e-8,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X_local, y[valid_mask])
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI) from global model
        mu_global, sigma_global = self.global_model.predict(X, return_std=True)
        sigma_global = np.clip(sigma_global, 1e-9, None)
        best_global = self.y.min()
        z = (best_global - mu_global) / sigma_global
        ei = (best_global - mu_global) * (1 - np.exp(-0.5 * z**2)) / (1 + z * np.sqrt(2 / np.pi)) + sigma_global * np.exp(-0.5 * z**2)
        
        # Lower Confidence Bound (LCB) from local model if available
        if self.local_model is not None:
            mu_local, sigma_local = self.local_model.predict(X, return_std=True)
            sigma_local = np.clip(sigma_local, 1e-9, None)
            kappa = 1.5
            lcb = mu_local - kappa * sigma_local
            # Weighted combination (70% global, 30% local)
            return 0.7 * ei + 0.3 * lcb
        else:
            return ei

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(200)  # Larger candidate pool for hybrid function
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)  # Minimize hybrid function
        top_candidates = candidates[idxs[:10]]  # More refinement points

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
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
        best_idx = np.argmin(acq_values_opt)  # Choose point with lowest hybrid value
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        y = np.zeros((n_new, 1))
        for i in range(n_new):
            y[i, 0] = func(X[i])
            if y[i, 0] < self.best_y:
                self.best_y = y[i, 0]
                self.best_x = X[i]
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
        # Initial Sobol design with maximin spacing
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.best_x = self.X[np.argmin(self.y)].copy()
        self.best_y = self.y.min()

        # Initialize global model
        self.global_model = self._fit_global_model(self.X, self.y)
        self.local_model = self._fit_local_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update both models
            self.global_model = self._fit_global_model(self.X, self.y)
            self.local_model = self._fit_local_model(self.X, self.y)

        return self.best_y, self.best_x
