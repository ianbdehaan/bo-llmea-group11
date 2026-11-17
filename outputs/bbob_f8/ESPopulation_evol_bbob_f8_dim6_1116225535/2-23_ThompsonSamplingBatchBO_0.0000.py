from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class ThompsonSamplingBatchBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Reduced initial design
        self.batch_size = 5
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = RBF(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e1))  # ARD RBF
        model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=2,  # Reduced restarts
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        samples = mu + sigma * self.rng.normal(size=mu.shape)  # Posterior sampling
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100)  # Increased candidate pool
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-20:]]  # 20 candidates for optimization

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            res = minimize(acq_wrapper, 
                           x0, 
                           bounds=[(l, u) for l, u in self.bounds], 
                           method='L-BFGS-B')
            if res.success:
                optimized_points.append(res.x)
        optimized_points = np.array(optimized_points)

        # Fill batch with top optimized points or fallback candidates
        if len(optimized_points) < batch_size:
            remaining = batch_size - len(optimized_points)
            optimized_points = np.vstack([optimized_points, top_candidates[:remaining]])

        # Select best batch_size points
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argsort(acq_values_opt, axis=0)[:, 0]
        return optimized_points[best_idx[:batch_size]]

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
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_points = self._select_next_points(self.batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)

            # Update best
            for i in range(len(new_y)):
                if new_y[i] < best_y:
                    best_y = new_y[i]
                    best_x = next_points[i]

            # Re-fit model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
