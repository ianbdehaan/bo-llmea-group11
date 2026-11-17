from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionLogEIBatchBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Increased initial sample size
        self.batch_size = 5   # Process 5 points per iteration
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.trust_width = 2.0  # Larger initial trust region
        self.trust_width_factor_expand = 1.1  # Smoother update factors
        self.trust_width_factor_contract = 0.9
        self.trust_width_min = 0.1
        self.trust_width_max = 3.0

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
            n_restarts_optimizer=1,  # Reduced from 3 to 1
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

    def _select_next_points(self):
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        lower = best_point - self.trust_width
        upper = best_point + self.trust_width
        lower = np.clip(lower, self.bounds[0], self.bounds[1])
        upper = np.clip(upper, self.bounds[0], self.bounds[1])

        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=50)  # More candidates for better coverage
        candidates = lower + (upper - lower) * candidates

        acq_vals = self._acquisition_function(candidates)
        top_indices = np.argsort(acq_vals)[-self.batch_size:]
        top_candidates = candidates[top_indices]

        optimized_points = []
        for x0 in top_candidates:
            result = optimize(
                lambda x: -self._acquisition_function(x[np.newaxis, :])[0],
                x0=x0,
                bounds=[(lower[i], upper[i]) for i in range(self.dim)],
                method='L-BFGS-B',
                options={'maxiter': 30}  # Reduced from 50 to 30
            )
            optimized_points.append(result.x)
        return np.array(optimized_points)

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
            next_points = self._select_next_points()
            new_y = self._evaluate_points(func, next_points)
            prev_best = np.min(self.y)
            self._update_eval_points(next_points, new_y)
            new_best = np.min(self.y)
            self.n_evals += self.batch_size

            if new_best < prev_best:
                self.trust_width = min(
                    self.trust_width * self.trust_width_factor_expand, 
                    self.trust_width_max
                )
            else:
                self.trust_width = max(
                    self.trust_width * self.trust_width_factor_contract, 
                    self.trust_width_min
                )

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
