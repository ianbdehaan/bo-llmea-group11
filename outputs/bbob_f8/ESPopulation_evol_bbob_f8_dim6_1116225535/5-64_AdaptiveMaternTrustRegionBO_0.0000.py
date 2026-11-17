from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import Bounds, minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class AdaptiveMaternTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim + 10  # Improved initialization size
        self.best_x = None
        self.best_y = float('inf')
        self.delta = 0.0
        self.tr_progress = 0  # Track progress for trust region updates
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,  # Reduced restarts for efficiency
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

        # Generate 100 random candidates in trust region
        candidates = self.rng.uniform(low=lower, high=upper, size=(100, self.dim))
        best_ei = -np.inf
        best_point = candidates[0]

        for x in candidates:
            x = x.reshape(1, -1)
            mu, sigma = self.model.predict(x, return_std=True)
            sigma = np.clip(sigma, 1e-9, None)
            improvement = mu - self.best_y
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
            ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
            if ei > best_ei:
                best_ei = ei
                best_point = x

        # Refine with L-BFGS-B
        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = self.model.predict(x, return_std=True)
            sigma = np.clip(sigma, 1e-9, None)
            improvement = mu - self.best_y
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
            ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
            return -ei.item()

        result = minimize(
            objective,
            x0=best_point,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 20, 'ftol': 1e-6}
        )
        next_point = result.x.reshape(1, -1) if result.success else best_point
        return next_point

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

        # Initialize trust region delta as median distance
        distances = np.linalg.norm(self.X - self.best_x, axis=1)
        self.delta = np.median(distances)
        self._adjust_delta()

        while self.n_evals < self.budget:
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.delta *= 1.3  # Moderate expansion
                self.tr_progress = 0  # Reset progress counter
            else:
                self.tr_progress += 1

            # Shrink trust region if no progress for 3 iterations
            if self.tr_progress >= 3:
                self.delta *= 0.7
                self._adjust_delta()

        return self.best_y, self.best_x

    def _adjust_delta(self):
        for i in range(self.dim):
            lower_available = 5.0 - self.best_x[i]
            upper_available = self.best_x[i] + 5.0
            available = min(lower_available, upper_available)
            self.delta = min(self.delta, available)
