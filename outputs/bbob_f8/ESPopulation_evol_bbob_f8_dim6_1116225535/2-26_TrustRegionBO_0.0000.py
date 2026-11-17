from collections.abc import Callable
from scipy.optimize import differential_evolution
from scipy.stats import norm
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Smaller initial Sobol design
        self.tr_radius = 2.0  # Initial trust region radius
        self.tr_decay = 0.9  # Radius decay factor per iteration
        self.min_tr_radius = 0.1
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, x):
        mu, sigma = self.model.predict(x.reshape(1, -1), return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        best_y = np.min(self.y)
        imp = mu - best_y
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei.item()  # Return negative for minimization

    def _select_next_point(self):
        def objective(x):
            return self._acquisition_function(x)
        bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=5,
            popsize=5,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=int(self.rng.integers(2**31 - 1))
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

    def _get_tr_points(self, best_x):
        dists = cdist(self.X, best_x.reshape(1, -1), metric='euclidean').ravel()
        mask = dists <= self.tr_radius
        return self.X[mask], self.y[mask]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x, best_y = self.X[best_idx], self.y[best_idx]

        # Initial trust region setup
        tr_X, tr_y = self._get_tr_points(best_x)
        self.model = self._fit_model(tr_X, tr_y)

        while self.n_evals < self.budget:
            next_point = self._select_next_point()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            current_y = np.min(self.y)
            if current_y < best_y:
                best_y = current_y
                best_x = self.X[np.argmin(self.y)]
                self.tr_radius = max(self.tr_radius * self.tr_decay, self.min_tr_radius)
            else:
                self.tr_radius = max(self.tr_radius * 1.1, self.min_tr_radius)  # Expand if no improvement

            # Refit model on trust region
            tr_X, tr_y = self._get_tr_points(best_x)
            self.model = self._fit_model(tr_X, tr_y)

        return best_y, best_x
