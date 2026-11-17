from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from numpy import random

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1  # Space-filling initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 1.0  # Initial trust region size
        self.radius_decay = 0.99  # Gradual shrinking factor
        self.model = None  # GP surrogate

    def _sample_points(self, n_points):
        # Generate initial points via Latin Hypercube sampling
        sampler = qmc.LatinHypercube(self.dim, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # GP with Matérn kernel for smoothness modeling
        kernel = C(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small nugget for stability
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0  # Reduce computational cost
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI)
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mean - best_y) / std
            ei = (mean - best_y) * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Trust region around current best solution
        current_best_idx = np.argmin(self.y)
        current_best = self.X[current_best_idx]
        lower = np.maximum(current_best - self.trust_radius, self.bounds[0])
        upper = np.minimum(current_best + self.trust_radius, self.bounds[1])

        # Ensure valid bounds for qmc.scale
        interval = upper - lower
        interval = np.maximum(interval, 1e-6)  # Avoid zero-length intervals
        lower = lower - interval * 0.0  # Ensure lower < upper in all dimensions

        # Generate diverse candidates within trust region
        sampler = qmc.LatinHypercube(self.dim, seed=self.rng)
        candidates = qmc.scale(sampler.random(20), lower, interval)

        best_candidate = None
        best_ei = -np.inf

        # Optimization of acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        bounds = [(l, u) for l, u in zip(lower, upper)]
        for x0 in candidates:
            res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.success and res.fun > best_ei:
                best_ei = res.fun
                best_candidate = res.x

        self.trust_radius *= self.radius_decay  # Adaptive shrinking
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate candidates without exceeding budget
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        # Maintain evaluated data
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.n_evals += initial_X.shape[0]
        self.model = self._fit_model(self.X, self.y)

        # Main optimization loop
        while self.n_evals < self.budget:
            batch_size = 1  # Sequential optimization
            new_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, new_points)
            self._update_eval_points(new_points, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += batch_size

        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()
