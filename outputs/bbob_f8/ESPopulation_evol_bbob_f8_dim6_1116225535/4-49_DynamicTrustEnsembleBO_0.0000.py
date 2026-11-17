from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class DynamicTrustEnsembleBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Increased initial design for better coverage
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 1.0  # Initial trust radius around best solution
        self.progress_count = 0   # Track consecutive progress for trust updates

    def _sample_points(self, n_points, use_global=True):
        """Hybrid sampling: global Sobol or local around best solution"""
        if use_global:
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            samples = sampler.random(n_points)
            scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
            return scaled_samples
        else:
            # Local candidates within trust region
            center = self.best_x
            radius = self.trust_radius
            candidates = center + radius * (self.rng.rand(n_points, self.dim) - 0.5)
            return np.clip(candidates, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        """Lightweight Random Forest with uncertainty estimation"""
        model = RandomForestRegressor(
            n_estimators=10,  # Balance accuracy and speed
            max_depth=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, f_best):
        """Expected Improvement with ensemble variance"""
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0)
        sigma = np.clip(sigma, 1e-9, None)
        imp = f_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[imp < 0] = 0.0  # No improvement if worse than best
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Hybrid acquisition: global and local candidates"""
        global_candidates = self._sample_points(40, use_global=True)
        local_candidates = self._sample_points(10, use_global=False)
        all_candidates = np.vstack([global_candidates, local_candidates])

        acq_values = self._acquisition_function(all_candidates, self.best_y)
        idxs = np.argsort(-acq_values, axis=0)[:, 0]  # Descending order
        top_candidates = all_candidates[idxs[:10]]  # Top 10 for optimization

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1), self.best_y).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return all_candidates[[0]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points, self.best_y)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        """Ensure budget is respected"""
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
        """Update data storage"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _update_trust_region(self):
        """Adjust trust region radius based on progress"""
        if self.best_y < self.last_best_y:  # Progress made
            self.progress_count += 1
            if self.progress_count >= 3:
                self.trust_radius = min(self.trust_radius * 1.2, 5.0)  # Expand
        else:  # No progress
            self.progress_count = 0
            if self.trust_radius > 1.0:
                self.trust_radius *= 0.5  # Contract
        self.last_best_y = self.best_y

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation with Latin Hypercube
        self.X = self._sample_points(self.n_init, use_global=True)
        self.y = self._evaluate_points(func, self.X)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]
        self.last_best_y = self.best_y

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Select next point
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best solution
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.progress_count = 0  # Reset progress counter on improvement
            else:
                self.progress_count += 1  # Track consecutive no-improvement

            # Adapt trust region
            self._update_trust_region()

            # Refit model
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
