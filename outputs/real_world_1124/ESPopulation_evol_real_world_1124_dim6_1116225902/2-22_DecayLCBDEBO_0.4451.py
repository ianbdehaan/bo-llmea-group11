from collections.abc import Callable
from scipy.optimize import differential_evolution
from scipy.stats import norm
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DecayLCBDEBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, dim + 1)  # More robust initial design for LCB
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None
        self.beta_initial = 5.0  # High initial exploration
        self.beta_decay_rate = 0.95  # Exponential decay for β

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

    def _compute_beta(self):
        """Decay β exponentially with evaluation progress."""
        decay_factor = self.beta_decay_rate ** (self.n_evals / self.budget)
        return self.beta_initial * decay_factor

    def _acquisition_function(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        beta = self._compute_beta()
        lcb = mean - beta * std  # Lower Confidence Bound for minimization
        return lcb.ravel()

    def _select_next_points(self, batch_size):
        # Define bounds for differential evolution
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        # Objective function to minimize (negative LCB)
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))
        # Run differential evolution with small population and limited iterations
        result = differential_evolution(
            objective, bounds, strategy='best1bin', maxiter=10, popsize=15,
            tol=1e-6, mutation=(0.5, 1), recombination=0.8, seed=self.rng
        )
        return result.x.reshape(1, -1)

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
            self.gp = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
