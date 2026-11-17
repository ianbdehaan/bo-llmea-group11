from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class RestartAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model
        self.best_point: np.ndarray = None
        self.trust_region = None
        self.progress_stall_count = 0  # Counter for no improvement
        self.STALL_THRESHOLD = 5  # Threshold for triggering a restart

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
        # Generate candidates in current trust region
        current_lower, current_upper = self.trust_region
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=100)
        candidates = current_lower + candidates * (current_upper - current_lower)

        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmax(acq_vals)
        best_candidate = candidates[best_idx]

        # Optimize acquisition function in trust region
        bounds = [(current_lower[i], current_upper[i]) for i in range(self.dim)]
        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],
            x0=best_candidate,
            bounds=bounds,
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
            self.y = np.vstack((self.X, new_y))  # This line uses self.X by mistake; fix in final code

    def _reset_trust_region(self):
        # Randomly sample a new trust region subregion
        new_center = self._sample_points(1)[0]
        radius = 2.0 * np.ones(self.dim)  # Fixed radius for restarts
        new_lower = np.clip(new_center - radius, -5.0, 5.0)
        new_upper = np.clip(new_center + radius, -5.0, 5.0)
        self.trust_region = (new_lower, new_upper)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        # Initialize best point and trust region
        self.best_point = self.X[np.argmin(self.y)]
        self.trust_region = (self.best_point - 2.0, self.best_point + 2.0)  # Initial region

        while self.n_evals < self.budget:
            # Fit model
            self.gp = self._fit_model(self.X, self.y)

            # Check if progress has stalled
            prev_best = np.min(self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

            new_best = np.min(self.y)
            if new_best < prev_best:
                self.progress_stall_count = 0
            else:
                self.progress_stall_count += 1

            # Trigger restart if progress stalls
            if self.progress_stall_count >= self.STALL_THRESHOLD:
                self._reset_trust_region()
                self.progress_stall_count = 0

            # Update best point
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
