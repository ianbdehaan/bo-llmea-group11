from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionLogEIAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.trust_width = 1.0
        self.trust_width_factor = 1.05
        self.trust_width_min = 0.1
        self.trust_width_max = 2.0

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        best_y = np.min(self.y)
        imp = best_y - mean
        imp = np.where(imp < 0, 0, imp)
        Z = imp / (std + 1e-8)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        log_ei = np.log(ei + 1e-10)
        return log_ei

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        lower = best_point - self.trust_width
        upper = best_point + self.trust_width
        lower = np.clip(lower, self.bounds[0], self.bounds[1])
        upper = np.clip(upper, self.bounds[0], self.bounds[1])

        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=10)
        candidates = lower + (upper - lower) * candidates

        acq_vals = self._acquisition_function(candidates)
        best_idx_candidate = np.argmax(acq_vals)
        best_candidate = candidates[best_idx_candidate]

        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],
            x0=best_candidate,
            bounds=[(lower[i], upper[i]) for i in range(self.dim)],
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        return result.x[np.newaxis, :]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            prev_best = np.min(self.y)
            self._update_eval_points(next_point, new_y)
            new_best = np.min(self.y)
            self.n_evals += 1

            if new_best < prev_best:
                self.trust_width = min(self.trust_width * self.trust_width_factor, self.trust_width_max)
            else:
                self.trust_width = max(self.trust_width / self.trust_width_factor, self.trust_width_min)

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
