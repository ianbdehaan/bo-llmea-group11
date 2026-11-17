from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
import warnings

class LocalHybridLHBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # More initial points for Latin Hypercube
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.max_batch_size = 2

    def _sample_points(self, n_points):
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_local_model(self, X, y):
        # Select top 20% of points based on function value
        top_idx = np.argsort(y, axis=0)[:max(5, int(0.2 * len(y)))]
        X_top = X[top_idx]
        y_top = y[top_idx].flatten()
        
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X_top, y_top)
        return model

    def _acquisition_function(self, X, iteration):
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        
        # Expected Improvement
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0
        
        # Lower Confidence Bound
        beta = max(0.0, 0.5 * (1 - iteration / self.budget))  # Dynamic exploration weight
        lcb = mean - beta * std
        
        # Hybrid acquisition (EI + scaled LCB)
        hybrid = ei - lcb  # Minimization: maximize EI, minimize LCB (negative for maximization)
        return hybrid

    def _select_next_points(self, batch_size):
        iteration = self.n_evals
        candidates = self._sample_points(100)  # Latin Hypercube candidates
        acq_vals = self._acquisition_function(candidates, iteration)
        top_indices = np.argsort(acq_vals)[-batch_size:]  # Select top candidates
        next_X = candidates[top_indices]
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Latin Hypercube
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial local model
        self.model = self._fit_local_model(self.X, self.y)
        
        batch_size = self.max_batch_size
        while self.n_evals < self.budget:
            # Generate and evaluate batch of points
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit local model with updated data
            self.model = self._fit_local_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
