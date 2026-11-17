from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class BatchEIGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube initial design for valley coverage
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = RBF(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-8,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        best_y = np.min(self.y) if self.y.size > 0 else np.inf
        imp = mu - best_y
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def _select_next_points(self, max_batch_size):
        remaining_budget = self.budget - self.n_evals
        batch_size = min(max_batch_size, remaining_budget)
        candidates = self._sample_points(200)  # LHS candidates for batch
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)
        top_candidates = candidates[idxs[-batch_size:]]  # Maximize EI
        return top_candidates

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
        # Initial Latin Hypercube design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize GP model
        self.model = self._fit_model(self.X, self.y)

        # Adaptive batch size schedule
        batch_schedule = np.linspace(3, 6, self.budget // 10).astype(int)

        while self.n_evals < self.budget:
            current_batch_size = batch_schedule[0] if batch_schedule.size > 0 else 5
            batch_schedule = batch_schedule[1:]  # Dequeue batch size
            next_points = self._select_next_points(current_batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)

            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            best_y = self.y[best_idx]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
