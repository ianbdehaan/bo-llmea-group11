from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class RandomForestVarianceBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 3 * dim)  # More diverse initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Latin Hypercube or Sobol samples."""
        if self.n_evals < self.n_init:
            samples = self.lhs_gen.random(n_points)
        else:
            samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low or self.bounds[0], high or self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Random Forest with uncertainty estimates."""
        model = RandomForestRegressor(
            n_estimators=50,  # Lightweight and fast for high-dimensional spaces
            criterion='squared_error',
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1  # Parallelize tree fitting
        )
        model.fit(X, y)
        return model

    def _predict_with_variance(self, X):
        """Compute mean and predictive variance (across tree predictions)."""
        y_pred = np.zeros((len(X), 1))
        for tree in self.model.estimators_:
            y_pred += tree.predict(X).reshape(-1, 1)
        y_pred /= len(self.model.estimators_)
        variance = np.var([tree.predict(X).reshape(-1, 1) for tree in self.model.estimators_], axis=0)
        return y_pred, variance

    def _acquisition_function(self, X):
        """Acquisition function: -mean * variance for minimization."""
        mean, variance = self._predict_with_variance(X)
        return -mean * variance  # Negative mean for minimization

    def _select_next_points(self):
        """Generate candidates and select top 10% based on acquisition."""
        n_candidates = max(100, 2 * self.dim)  # Ensure sufficient exploration
        candidates = self._sample_points(n_candidates)
        acq_vals = self._acquisition_function(candidates).flatten()
        top_n = max(5, int(0.1 * n_candidates))  # Select top 10% of candidates
        best_indices = np.argsort(acq_vals)[-top_n:]
        return candidates[best_indices]

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial Latin Hypercube sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
