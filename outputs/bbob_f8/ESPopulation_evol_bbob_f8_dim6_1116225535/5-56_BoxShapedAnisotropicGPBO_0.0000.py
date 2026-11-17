from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import differential_evolution, Bounds
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BoxShapedAnisotropicGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = dim + 1  # Minimal Latin Hypercube design
        self.best_x = None
        self.best_y = float('inf')
        self.length_scales = np.ones(dim)
        self.delta = 1.0  # Trust region scaling factor
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=1.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _get_trust_region_bounds(self):
        if self.X is None or self.best_x is None:
            return self.bounds
        # Get length scales from kernel
        self.length_scales = self.model.kernel_.get_params()['k2__length_scale']
        lower = np.clip(self.best_x - self.delta * self.length_scales, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + self.delta * self.length_scales, self.bounds[0], self.bounds[1])
        return np.vstack([lower, upper])

    def _select_next_points(self):
        bounds = self._get_trust_region_bounds()
        self.model = self._fit_model(self.X, self.y)

        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = self.model.predict(x, return_std=True)
            sigma = np.clip(sigma, 1e-9, None)
            improvement = mu - self.best_y
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
            ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
            return -ei.item()

        result = differential_evolution(
            objective,
            bounds=bounds.T,
            strategy='best1bin',
            maxiter=10,
            popsize=10,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.rng
        )
        if result.success:
            next_point = result.x.reshape(1, -1)
        else:
            next_point = self._sample_in_trust_region(1)
        return next_point

    def _sample_in_trust_region(self, n_points):
        bounds = self._get_trust_region_bounds()
        samples = self.rng.uniform(low=bounds[0], high=bounds[1], size=(n_points, self.dim))
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

        while self.n_evals < self.budget:
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.delta *= 1.5  # Expand trust region on progress
            else:
                self.delta = max(0.5, self.delta * 0.8)  # Contract if no progress

        return self.best_y, self.best_x
