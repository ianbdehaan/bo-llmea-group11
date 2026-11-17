from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DirectionalGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Budget-aware initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        # Track best trajectory for directional sampling
        self.best_trajectory = []

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _generate_directional_candidates(self, current_best, prev_best, num_directional=10):
        if not self.best_trajectory:
            return np.zeros((0, self.dim))  # No direction to follow initially
        # Generate points along the line connecting previous and current best
        t_values = np.linspace(0, 1, num_directional).reshape(-1, 1)
        directional_candidates = (1 - t_values) * prev_best + t_values * current_best
        return directional_candidates

    def _fit_model(self, X, y):
        kernel = Matern(nu=2.5)  # Anisotropic Matérn with ARD
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,  # Small noise
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, f_best):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)

        # Expected Improvement (EI)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = f_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[imp < 0] = 0.0  # No improvement if predicted worse than best
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Generate global candidates via Sobol
        global_candidates = self._sample_points(50)
        # Generate directional candidates from best trajectory
        if self.best_trajectory:
            prev_best = self.best_trajectory[-2]  # Previous best
            current_best = self.best_trajectory[-1]  # Current best
            directional_candidates = self._generate_directional_candidates(current_best, prev_best)
        else:
            directional_candidates = np.zeros((0, self.dim))
        # Combine and refine
        all_candidates = np.vstack([global_candidates, directional_candidates])
        acq_values = self._acquisition_function(all_candidates, self.best_y)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = all_candidates[idxs[:5]]  # Top 5 for optimization

        # Local refinement with L-BFGS-B
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
            return global_candidates[[0]].reshape(1, -1)
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
        self.best_trajectory.append(self.best_x)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]

            # Re-fit model
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
