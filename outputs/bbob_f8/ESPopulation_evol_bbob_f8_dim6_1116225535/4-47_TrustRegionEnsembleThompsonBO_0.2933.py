from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TrustRegionEnsembleThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Reduced initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.region_radius = 1.0  # Trust region half-size
        self.best_x = None

    def _sample_points(self, n_points, center=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            samples = sampler.random(n=n_points)
        else:
            # Generate within trust region
            lower = np.clip(center - self.region_radius, self.bounds[0], self.bounds[1])
            upper = np.clip(center + self.region_radius, self.bounds[0], self.bounds[1])
            bounds = np.stack([lower, upper])
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            samples = sampler.random(n=n_points)
            samples = qmc.scale(samples, bounds[0], bounds[1])
        return samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=10,  # Reduced from 20
            max_depth=5,  # Reduced from 10
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        sigma = np.clip(sigma, 1e-9, None)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(30, self.best_x)  # Reduced from 50
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-3:]]  # Reduced from 5 to 3

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            # Constrain to trust region
            lower = np.clip(self.best_x - self.region_radius, self.bounds[0], self.bounds[1])
            upper = np.clip(self.best_x + self.region_radius, self.bounds[0], self.bounds[1])
            bounds = [(l, u) for l, u in zip(lower, upper)]
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                self.best_x = next_point[0]
                self.region_radius *= 1.1  # Expand trust region on improvement
            else:
                self.region_radius *= 0.95  # Slight shrink otherwise
                self.region_radius = max(0.1, self.region_radius)  # Minimum radius

            self.model = self._fit_model(self.X, self.y)

        return best_y, self.best_x
