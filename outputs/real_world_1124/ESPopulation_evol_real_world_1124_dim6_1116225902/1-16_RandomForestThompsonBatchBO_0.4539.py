from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestThompsonBatchBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 10  # Initial Latin Hypercube samples
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))

    def _sample_points(self, n_points):
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Ensures single-core execution
        )
        model.fit(X, y.ravel())
        return model

    def _select_next_points(self, max_points):
        candidates = self._sample_points(200)  # Generate diverse candidates
        # Sample a tree from the Random Forest
        tree_idx = self.rng.integers(self.model.n_estimators)
        sampled_tree = self.model.estimators_[tree_idx]
        acq_vals = sampled_tree.predict(candidates)
        # Select top candidates for parallel evaluation
        best_indices = np.argsort(acq_vals)[:max_points]
        next_X = candidates[best_indices]
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
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        batch_size = 5  # Fixed batch size for parallelism
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals
            current_batch_size = min(remaining, batch_size)
            if current_batch_size <= 0:
                break
            next_X = self._select_next_points(current_batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
