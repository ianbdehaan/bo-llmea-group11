from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import Bounds, differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class AdaptiveLCBMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 10)  # Ensure at least 10 points
        self.best_x = None
        self.best_y = float('inf')
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.de_population = 20  # Initial DE population size
        self.de_maxiter = 50  # Max iterations for DE
        self.de_strategy = 'best1bin'

    def _sample_points(self, n_points):
        # Generate Sobol points with maximin greedy selection
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points * 2)  # Generate extra points for selection
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        indices = []
        while len(indices) < n_points:
            idx = self.rng.choice(len(scaled_samples))
            if idx not in indices:
                indices.append(idx)
            # Greedy maximin: remove points too close to selected ones
            distances = np.linalg.norm(scaled_samples - scaled_samples[idx], axis=1)
            mask = distances > 0.1
            scaled_samples = scaled_samples[mask]
        return scaled_samples[indices][:n_points]

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _select_next_points(self):
        model = self._fit_model(self.X, self.y)
        bounds = Bounds(self.bounds[0], self.bounds[1])

        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = model.predict(x, return_std=True)
            sigma = np.clip(sigma, 1e-9, None)
            # Dynamic exploration weight
            kappa = 1.0 / np.sqrt(self.n_evals + 1)
            lcb = mu - kappa * sigma
            return lcb.item()

        # Adjust DE parameters dynamically
        pop_size = max(10, int(self.de_population * 0.9))
        result = differential_evolution(
            func=objective,
            bounds=bounds,
            strategy=self.de_strategy,
            maxiter=self.de_maxiter,
            popsize=pop_size,
            mutation=(0.5, 1),
            recombination=0.7,
            tol=1e-6,
            seed=self.rng
        )

        if not result.success:
            next_x = self._sample_points(1)
        else:
            next_x = result.x.reshape(1, -1)

        return next_x

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
        best_idx = np.argmin(self.y)
        self.best_x = self.X[best_idx]
        self.best_y = self.y[best_idx]

        # Reduce DE population size for later iterations
        while self.n_evals < self.budget:
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]

            # Further reduce population size as budget decreases
            remaining = self.budget - self.n_evals
            if remaining < self.budget * 0.3:
                self.de_population = max(10, int(self.de_population * 0.7))

        return self.best_y, self.best_x
