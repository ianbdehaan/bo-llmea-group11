from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class TrustRegionEnsembleEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Larger initial design for trust region coverage
        self.trust_radius = 1.0  # Initial trust region radius (fraction of bounds)
        self.best_y = np.inf
        self.best_x = None
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.progress_counter = 0  # Counter for tracking progress
        self.max_progress_stalls = 5  # Max stalls before shrinking trust region

    def _sample_trust_points(self, center, radius):
        """Sample candidates within a trust region around center."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(50)  # Global candidates within trust region
        scaled_samples = qmc.scale(samples, center - radius, center + radius)
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, f_best):
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0)
        sigma = np.clip(sigma, 1e-9, None)

        imp = f_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[imp < 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, center, radius):
        candidates = self._sample_trust_points(center, radius)
        acq_values = self._acquisition_function(candidates, self.best_y)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[:5]]

        # Local refinement within trust region
        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1), self.best_y).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(c - radius, c + radius) for c in center],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[0]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points, self.best_y)
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Check progress and adjust trust region
            old_best = self.best_y
            next_point = self._select_next_points(self.best_x, self.trust_radius * 5.0)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0].copy()
                self.progress_counter = 0  # Reset progress counter
                if new_y[0] < old_best:  # Progress made
                    self.trust_radius *= 1.5  # Expand trust region
                    self.trust_radius = min(self.trust_radius, 5.0)  # Cap radius
            else:
                self.progress_counter += 1
                if self.progress_counter >= self.max_progress_stalls:
                    self.trust_radius *= 0.5  # Shrink trust region
                    self.trust_radius = max(self.trust_radius, 0.01)  # Minimum radius
                    self.progress_counter = 0
                    # Restart if region is too small
                    if self.trust_radius < 0.05:
                        print("Restarting due to small trust region")
                        self._restart_with_new_point(func)

            # Re-fit model
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x

    def _restart_with_new_point(self, func):
        """Restart the trust region with a new random point."""
        new_point = self._sample_points(1)
        new_y = self._evaluate_points(func, new_point)
        self._update_eval_points(new_point, new_y)
        if new_y[0] < self.best_y:
            self.best_y = new_y[0]
            self.best_x = new_point[0].copy()
        self.trust_radius = 1.0  # Reset trust region
        self.progress_counter = 0

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples
