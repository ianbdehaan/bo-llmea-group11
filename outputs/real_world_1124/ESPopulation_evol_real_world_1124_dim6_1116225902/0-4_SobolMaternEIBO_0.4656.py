from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class SobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.batch_size = min(10, self.budget - self.n_init)  # Adaptive batch size

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        return points * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=1.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            n_restarts_optimizer=0,  # Minimize hyperparameter optimization cost
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, y_min):
        mu, sigma = model.predict(X, return_std=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(imp) + sigma * norm.pdf(imp)
            ei[sigma == 0.] = 0.
        return ei

    def _select_next_points(self, model, y_min, batch_size):
        n_candidates = 200  # Reduce computational overhead
        candidates = self._sample_points(n_candidates)
        ei = self._acquisition_function(candidates, model, y_min)
        idx = np.argsort(-ei, axis=0)[:batch_size]
        return candidates[idx]

    def _evaluate_points(self, func, X):
        results = np.array([func(x) for x in X])
        return results.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        self.n_evals = self.n_init

        # Main optimization loop
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            y_min = np.min(self.y)
            remaining = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining)
            next_points = self._select_next_points(model, y_min, batch_size)
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            self.n_evals += batch_size

        # Return best solution
        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
