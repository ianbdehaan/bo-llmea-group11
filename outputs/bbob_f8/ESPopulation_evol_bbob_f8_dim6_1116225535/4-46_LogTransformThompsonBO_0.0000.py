from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class LogTransformThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None  # log-transformed y
        self.original_y: np.ndarray = None  # original y for tracking best
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)  # Avoid division by zero
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(50)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-5:]]  # Last 5 (highest acq values)

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
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
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        original_y = np.zeros((n_new, 1))
        for i in range(n_new):
            original_y[i, 0] = func(X[i])
        # Add small epsilon to avoid log(0)
        log_y = np.log(original_y + 1e-12)
        self.n_evals += n_new
        return log_y, original_y

    def _update_eval_points(self, new_X, new_log_y, new_original_y):
        if self.X is None:
            self.X = new_X
            self.y = new_log_y
            self.original_y = new_original_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_log_y])
            self.original_y = np.vstack([self.original_y, new_original_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        log_y, original_y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, log_y, original_y)
        best_y = np.min(original_y)
        best_x = self.X[np.argmin(original_y)]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_log_y, new_original_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_log_y, new_original_y)
            
            if new_original_y[0] < best_y:
                best_y = new_original_y[0]
                best_x = next_point[0]
            
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
