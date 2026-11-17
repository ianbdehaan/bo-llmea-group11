from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class LocalSurrogateHybridACQBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 2 * dim)  # Latin hypercube initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Select top 10% of points for local modeling
        n_best = max(5, int(0.1 * len(y)))
        best_idx = np.argsort(y)[:n_best]
        X_best = X[best_idx]
        y_best = y[best_idx]
        
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)  # Anisotropic Matern kernel
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X_best, y_best)
        return model

    def _acquisition_function(self, X, current_best, progress_ratio):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        
        # Expected Improvement (EI)
        best_value = current_best
        with np.errstate(divide='ignore', invalid='ignore'):
            ei = (best_value - mu) * scipy.stats.norm.cdf((best_value - mu) / sigma)
            ei[sigma == 0] = 0.0
        
        # Distance penalty to current best point
        current_best_point = self.X[np.argmin(self.y)]
        distances = np.linalg.norm(X - current_best_point, axis=1)
        penalty_weight = np.exp(-10 * progress_ratio)  # Exponentially decaying penalty
        penalty = penalty_weight * distances
        
        return ei - penalty  # Minimize hybrid function

    def _select_next_points(self, current_best, progress_ratio):
        bounds = np.array([[-5.0]*self.dim, [5.0]*self.dim]).T
        
        # Define acquisition objective for differential evolution
        def objective(X):
            return -self._acquisition_function(X.reshape(1, -1), current_best, progress_ratio).item()
        
        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=10,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=self.rng.integers(2**31 - 1)
        )
        
        return result.x.reshape(1, -1)

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
        # Initial Latin Hypercube design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize local GP model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            progress_ratio = self.n_evals / self.budget
            next_point = self._select_next_points(best_y, progress_ratio)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
