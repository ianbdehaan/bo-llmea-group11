from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize as optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DynamicTrustRegionAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 3 * dim)  # Enhanced initial sampling
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp = None  # Surrogate model
        self.best_point: np.ndarray = None
        self.trust_radius: np.ndarray = None
        self.no_improvement_count = 0  # Counter for no improvement steps

    def _sample_points(self, n_points):
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sobol.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        # Conditional hyperparameter optimization: only retrain if data improved significantly
        if self.gp is None or (self.y.shape[0] > 10 and np.random.rand() < 0.2):
            # Create Matérn kernel with ARD (anisotropic)
            kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,
                                  length_scale_bounds=(1e-2, 1e2))
            # Fit Gaussian Process with reduced restarts
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=2,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp
        return self.gp  # Reuse previous model if data not sufficient or no significant improvement

    def _acquisition_function(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        best_y = np.min(self.y)
        imp = best_y - mean
        imp = np.where(imp < 0, 0, imp)
        Z = imp / (std + 1e-8)
        # Expected improvement
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        # LogEI for optimization
        log_ei = np.log(ei + 1e-10)
        return log_ei

    def _select_next_points(self, batch_size):
        # Generate candidate points within trust region
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        # Generate 50 Sobol samples in unit cube
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=50)  # Increased candidate count
        candidates = current_lower + candidates * (current_upper - current_lower)

        # Evaluate acquisition function for candidates
        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmax(acq_vals)
        best_candidate = candidates[best_idx]

        # Optimize the acquisition function within the trust region
        bounds = [(current_lower[i], current_upper[i]) for i in range(self.dim)]
        result = optimize(
            lambda x: -self._acquisition_function(x[np.newaxis, :])[0],  # Maximize by minimizing negative
            x0=best_candidate,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 30}  # Reduced iterations for efficiency
        )
        return result.x[np.newaxis, :]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]

        # Initialize best point and trust radius
        self.best_point = self.X[np.argmin(self.y)]
        self.trust_radius = np.ones(self.dim) * 3.0  # Reduced initial radius for faster convergence

        while self.n_evals < self.budget:
            # Fit model
            self.gp = self._fit_model(self.X, self.y)

            # Select next point
            next_point = self._select_next_points(1)

            # Evaluate
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1

            # Check improvement and adjust trust region
            prev_best = np.min(self.y)
            new_best = np.min(self.y)
            if new_best < prev_best:
                # Significant improvement: expand trust radius
                self.trust_radius = np.clip(self.trust_radius * 1.2, 0.1, 5.0)
                self.no_improvement_count = 0  # Reset counter
            else:
                # No improvement: shrink trust radius and increment counter
                self.trust_radius = np.clip(self.trust_radius * 0.8, 0.1, 5.0)
                self.no_improvement_count += 1

            # Trust region reset after 3 no-improvement steps
            if self.no_improvement_count >= 3:
                self.trust_radius = np.clip(self.trust_radius * 2.0, 0.1, 5.0)
                self.no_improvement_count = 0

            # Update best point and trust center
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
