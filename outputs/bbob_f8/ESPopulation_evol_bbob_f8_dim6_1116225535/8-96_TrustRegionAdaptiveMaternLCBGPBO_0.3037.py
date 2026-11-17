from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionAdaptiveMaternLCBGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 5 * dim  # Expanded initialization for better coverage
        self.batch_size = 5
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 2.0  # Initial trust region radius

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)  # Anisotropic Matérn kernel with ν=5/2
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # More robust hyperparameter optimization
            alpha=1e-8,  # Reduced noise assumption for deterministic function
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        current_kappa = self._compute_kappa()
        return mu - current_kappa * sigma  # Minimize LCB

    def _compute_kappa(self):
        progress_ratio = self.n_evals / self.budget
        return 1.5 * np.exp(-2 * progress_ratio)  # Exponential decay of exploration weight

    def _select_next_points(self, batch_size):
        # Expand bounds based on trust region
        best_x = self.X[np.argmin(self.y)]
        trust_bounds = np.array([best_x - self.trust_radius, best_x + self.trust_radius])
        trust_bounds = np.clip(trust_bounds, self.bounds[0], self.bounds[1])  # Stay within global bounds

        candidates = self._sample_points(250, bounds=trust_bounds)  # Increased candidate pool
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)  # Minimize LCB
        top_candidates = candidates[idxs[:25]]  # Top 25 candidates for refinement

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(trust_bounds[0], trust_bounds[1])]
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[idxs[:batch_size]]
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_indices = np.argsort(acq_values_opt)[:batch_size]
        return optimized_points[best_indices]

    def _update_trust_region(self, new_X, new_y):
        best_x = self.X[np.argmin(self.y)]
        best_new_x = new_X[np.argmin(new_y)]
        # Adjust trust region based on progress
        if new_y.min() < self.y.min():
            # Successful step: expand trust region
            self.trust_radius = min(self.trust_radius * 1.5, 2.0)
        else:
            # No progress: shrink trust region
            self.trust_radius = max(self.trust_radius * 0.5, 0.5)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        remaining_budget = self.budget - self.n_evals
        if n_new > remaining_budget:
            n_new = remaining_budget
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
        # Initial Sobol design for diverse sampling
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize GP model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_points = self._select_next_points(self.batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)

            # Update best solution
            for i in range(len(new_y)):
                if new_y[i] < best_y:
                    best_y = new_y[i]
                    best_x = next_points[i]

            # Update trust region
            self._update_trust_region(next_points, new_y)

            # Refit GP model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
