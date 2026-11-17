from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional

class AdaptiveTrustRegionThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = self.dim + 1
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.refit_interval = 5  # Only retrain every 5 evaluations

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=10,  # Reduced from 20
            max_depth=5,      # Reduced from 10
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Use a single Thompson sample from the ensemble
        idx = self.rng.integers(len(self.model.estimators_))
        return self.model.estimators_[idx].predict(X).reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Generate fewer candidates and optimize top candidates
        n_candidates = 20  # Reduced from 50
        candidates = self._sample_points(n_candidates, bounds=self.trust_bounds)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-3:]]  # Optimized top 3 candidates

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(self.trust_bounds[0], self.trust_bounds[1])]
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

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

    def _update_trust_bounds(self, success: bool, trust_radius: float, best_x: np.ndarray) -> Tuple[float, np.ndarray]:
        # Adaptive scaling based on success rate over last 5 steps
        recent_successes = np.mean(self.success_history[-5:]) if len(self.success_history) >= 5 else success
        if success:
            scale = 1.3 if recent_successes >= 0.7 else 1.1
        else:
            scale = 0.7 if recent_successes < 0.3 else 0.9
        new_radius = trust_radius * scale
        new_bounds = np.array([
            np.clip(best_x - new_radius, -5.0, 5.0),
            np.clip(best_x + new_radius, -5.0, 5.0)
        ])
        return new_radius, new_bounds

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> Tuple[np.float64, np.ndarray]:
        # Initialize with Latin Hypercube
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize trust region with moving average of best points
        self.best_points = [best_x.copy()]
        self.success_history = []
        trust_radius = 1.0
        self.trust_bounds = np.array([
            np.clip(best_x - trust_radius, -5.0, 5.0),
            np.clip(best_x + trust_radius, -5.0, 5.0)
        ])
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best point and success history
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
                success = True
            else:
                success = False

            self.success_history.append(success)
            self.best_points.append(best_x.copy())
            if len(self.best_points) > 5:
                self.best_points.pop(0)

            # Update trust bounds using adaptive scaling
            trust_radius, new_bounds = self._update_trust_bounds(
                success, trust_radius, 
                np.mean(self.best_points, axis=0)  # Use moving average of best points
            )
            self.trust_bounds = new_bounds

            # Retrain model less frequently
            if (self.n_evals % self.refit_interval) == 0:
                self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
