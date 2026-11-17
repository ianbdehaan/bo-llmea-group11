from collections.abc import Callable
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math

class DynamicTrustRegionDEBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.model: GaussianProcessRegressor = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Smaller initial Sobol design
        self.trust_radius = 3.0  # Initial trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            alpha=1e-8,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, best_y):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        improvements = np.clip(best_y - mu, 0, None)
        Z = improvements / sigma
        ei = improvements * np.exp(-0.5 * Z**2) / (sigma * np.sqrt(2 * np.pi))
        return ei  # Maximize EI

    def _select_next_point(self, best_x, best_y):
        def objective(x):
            x = x.reshape(1, -1)
            return -self._acquisition_function(x, best_y).item()  # Minimize negative EI

        # Define trust region bounds
        lower = np.clip(best_x - self.trust_radius, self.bounds[0], self.bounds[1])
        upper = np.clip(best_x + self.trust_radius, self.bounds[0], self.bounds[1])
        trust_bounds = [(l, u) for l, u in zip(lower, upper)]

        # Generate DE population (20 candidates) with mix of global and local points
        pop_size = 20
        global_points = self._sample_points(pop_size // 2, bounds=self.bounds)
        local_points = self._sample_points(pop_size // 2, bounds=[lower, upper])
        population = np.vstack([global_points, local_points])

        result = differential_evolution(
            objective,
            bounds=trust_bounds,
            strategy='best1bin',
            popsize=pop_size,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            maxiter=50,
            polish=False,
            seed=self.rng.integers(2**31 - 1)
        )
        return result.x.reshape(1, -1)

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
        # Initial Sobol design with smaller size
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize GP model with global data
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Select next point using DE within trust region
            next_point = self._select_next_point(best_x, best_y)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            # Shrink trust region
            self.trust_radius *= 0.8

            # Re-fit model with only local points
            local_mask = np.linalg.norm(self.X - best_x, axis=1) <= self.trust_radius
            local_X, local_y = self.X[local_mask], self.y[local_mask]
            if len(local_X) < self.dim + 1:  # Ensure enough points for GP
                local_X, local_y = self.X, self.y  # Fall back to global if insufficient local data
            self.model = self._fit_model(local_X, local_y)

        return best_y, best_x
