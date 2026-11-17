from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize, differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class LocalGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Efficient global initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Find points near best solution
        dists = np.linalg.norm(X - self.best_x, axis=1)
        radius = np.sqrt(self.dim) * 1.0  # Adaptive radius
        near_mask = dists < radius
        X_near = X[near_mask]
        y_near = y[near_mask]
        if len(X_near) == 0:
            X_near = X
            y_near = y

        # Anisotropic Matérn kernel with ARD
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise for stability
            n_restarts_optimizer=1,  # Limit computational cost
            optimizer="fmin_l_bfgs_b",
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X_near, y_near)
        return model

    def _acquisition_function(self, X, f_best):
        mean, std = self.model.predict(X, return_std=True)
        std = np.clip(std, 1e-9, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = f_best - mean
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[imp < 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        def acq_func(x):
            return -self._acquisition_function(x.reshape(1, -1), self.best_y).item()

        result = differential_evolution(
            acq_func,
            bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
            strategy="best1bin",
            maxiter=5,  # Budget-aware optimization
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.8,
            seed=self.rng.integers(2**31 - 1)
        )
        if result.success:
            return result.x.reshape(1, -1)
        else:
            return self._sample_points(1)

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
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]

            # Re-fit model
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
