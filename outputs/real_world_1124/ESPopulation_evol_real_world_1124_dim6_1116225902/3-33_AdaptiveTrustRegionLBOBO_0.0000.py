from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class AdaptiveTrustRegionLBOBO:
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
        self.best_point: np.ndarray = None
        self.trust_radius: float = 5.0  # Initial radius for all dimensions

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

    def _select_next_points(self):
        # Trust region bounds
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)

        # Sample a function from the GP posterior
        noise = self.rng.normal()  # Single noise term for all x

        def sample_function(x):
            x = x.reshape(-1, self.dim)
            mean, std = self.gp.predict(x, return_std=True)
            return mean.flatten() + noise * std.flatten()

        # Generate 10 Sobol starting points within trust region
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        initial_points = current_lower + sobol.random(n=10) * (current_upper - current_lower)

        best_x = None
        best_value = np.inf

        for x0 in initial_points:
            # Define bounds for L-BFGS-B
            bounds = [(l, u) for l, u in zip(current_lower, current_upper)]
            # Optimize the sample function with gradient-based method
            res = minimize(
                fun=lambda x: sample_function(x),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-9, 'maxiter': 50}
            )
            if res.success and res.fun < best_value:
                best_value = res.fun
                best_x = res.x

        return best_x[np.newaxis, :]

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
        self.best_point = self.X[np.argmin(self.y)]
        self.trust_radius = 5.0  # Initial trust radius

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

            # Trust region update
            prev_best = np.min(self.y)
            self.best_point = self.X[np.argmin(self.y)]
            if np.min(self.y) < prev_best:
                self.trust_radius = min(self.trust_radius * 1.1, 5.0)  # Expand on progress
            else:
                self.trust_radius = max(self.trust_radius * 0.9, 0.1)  # Contract on stall

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
