from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics.pairwise import euclidean_distances

class LocalGPThompsonDifferentialBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = dim + 2  # Reduced initial sampling for efficiency
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.trust_radius = 1.0
        self.trust_radius_factor = 1.2  # Adaptively expand/shrink the trust region
        self.trust_radius_min = 0.5
        self.trust_radius_max = 3.0

    def _sample_points(self, n_points):
        lhs = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = lhs.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                              length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _get_local_points(self, best_point, radius):
        distances = euclidean_distances(self.X, [best_point]).flatten()
        mask = distances <= radius
        return self.X[mask], self.y[mask]

    def _acquisition_function(self, x, gp):
        return -gp.predict(x[np.newaxis, :])[0]  # Maximize negative prediction

    def _select_next_point(self):
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        local_X, local_y = self._get_local_points(best_point, self.trust_radius)

        # Refit GP with local data
        self.gp = self._fit_model(local_X, local_y)

        # Define bounds for differential evolution within trust region
        lower = np.clip(best_point - self.trust_radius, self.bounds[0], self.bounds[1])
        upper = np.clip(best_point + self.trust_radius, self.bounds[0], self.bounds[1])

        # Optimize sampled function via differential evolution
        bounds = [(l, u) for l, u in zip(lower, upper)]
        result = differential_evolution(
            self._acquisition_function,
            bounds=bounds,
            args=(self.gp,),
            strategy='best1bin',
            maxiter=10,
            popsize=8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=self.rng
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
            next_point = self._select_next_point()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

            # Update trust radius based on progress
            prev_best = np.min(self.y[:-1])
            new_best = np.min(self.y)
            if new_best < prev_best:
                self.trust_radius = min(self.trust_radius * self.trust_radius_factor, self.trust_radius_max)
            else:
                self.trust_radius = max(self.trust_radius / self.trust_radius_factor, self.trust_radius_min)

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
