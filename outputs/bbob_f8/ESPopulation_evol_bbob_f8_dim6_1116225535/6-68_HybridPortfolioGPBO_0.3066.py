from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class HybridPortfolioGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Maximin Sobol initialization
        self.best_x = None
        self.best_y = float('inf')
        self.delta = 1.0  # Adaptive trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.evolution_restarts = 3
        self.acq_weights = np.array([0.7, 0.3])  # EI: LogEI

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _evaluate_acq(self, x, model, best_y):
        x = x.reshape(1, -1)
        mu, sigma = model.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)

        # Expected Improvement
        improvement = mu - best_y
        z = improvement / sigma
        ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
        ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)

        # Log Expected Improvement
        log_ei = np.log(ei + 1e-12)
        return self.acq_weights[0] * ei + self.acq_weights[1] * log_ei

    def _evolution_strategy(self, model, bounds, best_y):
        x = np.random.uniform(bounds[0], bounds[1], size=self.dim)
        for _ in range(20):  # Small ES
            offspring = x + self.rng.normal(0, 0.1 * self.delta, size=self.dim)
            offspring = np.clip(offspring, bounds[0], bounds[1])
            acq_current = self._evaluate_acq(x, model, best_y)
            acq_offspring = self._evaluate_acq(offspring, model, best_y)
            if acq_offspring > acq_current:
                x = offspring
        return x

    def _select_next_points(self, model, bounds, best_y):
        candidates = [self._evolution_strategy(model, bounds, best_y) for _ in range(self.evolution_restarts)]
        acqs = [self._evaluate_acq(c, model, best_y) for c in candidates]
        best_idx = np.argmax(acqs)
        return candidates[best_idx].reshape(1, -1)

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

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Adaptive trust region based on improvement variance
            recent_improvements = self.y[-min(5, len(self.y)):] - self.best_y
            self.delta = 1.0 + 2.0 * np.std(recent_improvements)
            self.delta = max(min(self.delta, 3.0), 0.1)  # Clamp delta

            # Trust region bounds
            lower = np.clip(self.best_x - self.delta, self.bounds[0], self.bounds[1])
            upper = np.clip(self.best_x + self.delta, self.bounds[0], self.bounds[1])
            bounds = (lower, upper)

            next_point = self._select_next_points(self.model, bounds, self.best_y)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
                self.acq_weights = np.array([0.9, 0.1])  # Shift toward EI
            else:
                self.acq_weights = np.array([0.5, 0.5])  # Balance

        return self.best_y, self.best_x
