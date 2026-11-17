from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = dim + 1  # Latin Hypercube for initial coverage
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.tr_radius = 2.0  # Initial trust region radius
        self.tr_center = np.zeros(dim)  # Centered at origin initially
        self.success_count = 0  # Track progress for trust region adaptation

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        beta = 0.5 * np.exp(-0.01 * self.n_evals)  # Dynamic exploration
        acq = mu - beta * sigma
        return acq.reshape(-1, 1)

    def _clip_to_trust_region(self, x):
        # Project x into the trust region bounds [center - radius, center + radius]
        lower = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        upper = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        return np.clip(x, lower, upper)

    def _select_next_point(self):
        # Sample 30 candidates uniformly within trust region
        candidates = (self.rng.rand(30, self.dim) * 2 * self.tr_radius - self.tr_radius) + self.tr_center
        candidates = self._clip_to_trust_region(candidates)

        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[:5]]  # Top 5 candidates for local opt

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                x_tr = self._clip_to_trust_region(x)
                return -self._acquisition_function(x_tr.reshape(1, -1)).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[0]].reshape(1, -1)
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

    def _adjust_trust_region(self, new_x, improvement):
        """Adapt trust region based on progress."""
        # Shrink or expand based on improvement and success count
        if improvement > 0:
            self.success_count += 1
            if self.success_count >= 3:  # Expand after 3 successful improvements
                self.tr_radius *= 1.2
                self.success_count = 0
        else:
            self.success_count = 0
            self.tr_radius *= 0.7  # Shrink if no improvement
        # Ensure radius doesn't exceed bounds
        self.tr_radius = np.clip(self.tr_radius, 0.1, 2.5)
        # Update trust region center to the new best point
        self.tr_center = new_x

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_point()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            
            # Update best
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
                improvement = best_y_prev - best_y if hasattr(self, 'best_y_prev') else best_y
                best_y_prev = best_y
                self._adjust_trust_region(next_point[0], improvement)
            else:
                self._adjust_trust_region(next_point[0], 0)

            # Only re-fit model when trust region changes
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
