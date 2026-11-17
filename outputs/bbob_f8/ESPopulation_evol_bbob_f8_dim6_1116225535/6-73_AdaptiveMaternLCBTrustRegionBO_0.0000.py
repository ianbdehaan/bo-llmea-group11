from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import Bounds, minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class AdaptiveMaternLCBTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim  # Larger initial Sobol design
        self.best_x = None
        self.best_y = float('inf')
        self.delta = 0.0  # Trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.candidates_per_iter = 50  # Number of Sobol candidates
        self.lbfgs_max_iter = 50  # Reduced L-BFGS iterations
        self.kappa = 2.0  # Initial LCB exploration parameter
        self.kappa_decay = 0.995  # Per-iteration decay for κ

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,  # Reduced from 3 for speed
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _get_points_in_trust_region(self):
        if self.X is None or self.best_x is None:
            return self.X, self.y
        distances = np.linalg.norm(self.X - self.best_x, axis=1)
        mask = distances <= self.delta
        return self.X[mask], self.y[mask]

    def _select_next_points(self):
        X_trust, y_trust = self._get_points_in_trust_region()
        self.model = self._fit_model(X_trust, y_trust)

        lower = np.clip(self.best_x - self.delta, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + self.delta, self.bounds[0], self.bounds[1])
        bounds = Bounds(lower, upper)

        # Generate Sobol candidates within trust region
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(self.candidates_per_iter)
        scaled_samples = qmc.scale(samples, lower, upper)
        candidates = scaled_samples

        best_lcb = -np.inf
        best_x = None

        for candidate in candidates:
            result = minimize(
                fun=self._objective_lcb,
                x0=candidate,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.lbfgs_max_iter, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            if result.success:
                lcb = -result.fun  # Negative LCB for maximization
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_x = result.x

        if best_x is None:  # Fallback
            best_x = self._sample_in_trust_region(1).flatten()

        return np.array([best_x])

    def _objective_lcb(self, x):
        x = x.reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        lcb = mu - self.kappa * sigma
        lcb = np.nan_to_num(lcb, neginf=-np.inf, posinf=np.inf)
        return -lcb.item()

    def _sample_in_trust_region(self, n_points):
        lower = np.clip(self.best_x - self.delta, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + self.delta, self.bounds[0], self.bounds[1])
        samples = self.rng.uniform(low=lower, high=upper, size=(n_points, self.dim))
        return samples

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
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        self.best_x = self.X[best_idx]
        self.best_y = self.y[best_idx]

        if len(self.X) > 1:
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            self.delta = max(distances)  # Initial trust region size
        else:
            self.delta = 1.0
        self._adjust_delta()

        while self.n_evals < self.budget:
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.delta *= 1.2  # Expand trust region on improvement
            else:
                self.delta *= 0.8  # Contract on no improvement

            self._adjust_delta()
            self.kappa *= self.kappa_decay  # Decay exploration parameter

        return self.best_y, self.best_x

    def _adjust_delta(self):
        delta_min = float('inf')
        for i in range(self.dim):
            lower_available = 5.0 - self.best_x[i]
            upper_available = self.best_x[i] + 5.0
            available = min(lower_available, upper_available)
            delta_min = min(delta_min, available)
        self.delta = min(self.delta, delta_min)
