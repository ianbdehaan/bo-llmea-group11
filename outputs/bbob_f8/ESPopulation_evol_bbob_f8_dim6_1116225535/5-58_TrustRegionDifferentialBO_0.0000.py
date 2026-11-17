from collections.abc import Callable
from scipy.optimize import Bounds
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class TrustRegionDifferentialBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Smaller initial design
        self.best_x = None
        self.best_y = float('inf')
        self.delta = 0.0  # Trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.pop_size = 20  # Differential evolution population size
        self.generations = 10  # Generations per acquisition step
        self.F = 0.9  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def _sample_points(self, n_points):
        samples = self.rng.uniform(low=-5.0, high=5.0, size=(n_points, self.dim))
        return samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(self.dim))
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _get_points_in_trust_region(self):
        if self.X is None or self.best_x is None:
            return self.X, self.y
        distances = np.linalg.norm(self.X - self.best_x, axis=1)
        mask = distances <= self.delta
        return self.X[mask], self.y[mask]

    def _select_next_points(self):
        X_trust, y_trust = self._get_points_in_trust_region()
        self.model = self._fit_model(X_trust, y_trust)

        lower = np.clip(self.best_x - self.delta, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + self.delta, self.bounds[0], self.bounds[1])

        # Initialize DE population
        population = self.rng.uniform(low=lower, high=upper, size=(self.pop_size, self.dim))
        bounds = Bounds(lower, upper)

        for gen in range(self.generations):
            for i in range(self.pop_size):
                # Select 3 distinct indices
                idxs = self.rng.choice(self.pop_size, size=3, replace=False)
                a, b, c = population[idxs]
                # Mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lower, upper)
                # Crossover
                crossover_mask = self.rng.random(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                # Evaluate
                ei_mutant = self._objective_ei(population[i])
                ei_trial = self._objective_ei(trial)
                # Replace if trial improves EI
                if ei_trial > ei_mutant:
                    population[i] = trial

        # Select best candidate from final population
        ei_values = np.array([self._objective_ei(x) for x in population])
        best_idx = np.argmax(ei_values)
        return np.array([population[best_idx]])

    def _objective_ei(self, x):
        x = x.reshape(1, -1)
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        improvement = mu - self.best_y
        z = improvement / sigma
        ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
        return ei.item()

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

        if len(self.X) > 1:
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            self.delta = max(distances)
        else:
            self.delta = 1.0
        self._adjust_delta()

        while self.n_evals < self.budget:
            next_point = self._select_next_points()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.delta *= 1.5
                self._adjust_delta()

        return self.best_y, self.best_x

    def _adjust_delta(self):
        delta_min = float('inf')
        for i in range(self.dim):
            lower_available = 5.0 - self.best_x[i]
            upper_available = self.best_x[i] + 5.0
            available = min(lower_available, upper_available)
            delta_min = min(delta_min, available)
        self.delta = min(self.delta, delta_min)
