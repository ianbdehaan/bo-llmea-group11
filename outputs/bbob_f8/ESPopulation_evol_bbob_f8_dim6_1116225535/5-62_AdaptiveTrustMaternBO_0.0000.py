from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import Bounds, minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class AdaptiveTrustMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Initial Sobol design for low-discrepancy coverage
        self.best_x = None
        self.best_y = float('inf')
        self.delta = 0.0  # Trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.candidates_per_iter = 30  # Reduced candidates with Sobol sequences
        self.lbfgs_max_iter = 50  # Reduced L-BFGS-B iterations per candidate

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,  # Reduced for speed
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

        # Evaluate candidates and select top 10 for refinement
        candidates = scaled_samples
        ei_values = np.array([self._ei(candidate) for candidate in candidates])
        top_indices = np.argsort(ei_values)[-10:]  # Top 10 candidates
        top_candidates = candidates[top_indices]

        best_ei = -np.inf
        best_x = None

        for candidate in top_candidates:
            result = minimize(
                fun=self._objective_ei,
                x0=candidate,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.lbfgs_max_iter, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            if result.success:
                ei = -result.fun
                if ei > best_ei:
                    best_ei = ei
                    best_x = result.x

        if best_x is None:
            best_x = self._sample_in_trust_region(1).flatten()

        return np.array([best_x])

    def _ei(self, x):
        x = x.reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        improvement = mu - self.best_y
        z = improvement / sigma
        ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
        return ei.item()

    def _objective_ei(self, x):
        x = x.reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        improvement = mu - self.best_y
        z = improvement / sigma
        ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
        ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
        return -ei.item()

    def _sample_in_trust_region(self, n_points):
        lower = np.clip(self.best_x - self.delta, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + self.delta, self.bounds[0], self.bounds[1])
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), lower, upper)
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
            self.delta = max(distances)
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
                self.delta *= 1.25  # Conservative expansion
                self._adjust_delta()
            else:
                self.delta *= 0.9  # Contraction

        return self.best_y, self.best_x

    def _adjust_delta(self):
        delta_min = float('inf')
        for i in range(self.dim):
            lower_available = 5.0 - self.best_x[i]
            upper_available = self.best_x[i] + 5.0
            available = min(lower_available, upper_available)
            delta_min = min(delta_min, available)
        self.delta = min(self.delta, delta_min)
        # Adjust based on model uncertainty
        mu, sigma = self.model.predict(self.X, return_std=True)
        avg_variance = np.mean(sigma)
        if avg_variance > 1e-3:
            self.delta = max(self.delta, 0.5)  # Expand cautiously in high-uncertainty regions
        else:
            self.delta = max(self.delta, 0.1)  # Shrink in low-uncertainty regions
