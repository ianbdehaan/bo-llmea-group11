from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GeneticEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Latin Hypercube initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.best_y = np.inf
        self.best_x = None

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        y_min = self.best_y
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - y_min
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[imp < 0] = 0.0  # No improvement
        return ei

    def _select_next_points(self, batch_size):
        pop_size = 20
        generations = 2
        bounds = self.bounds
        population = self._sample_points(pop_size, bounds=bounds)
        
        for _ in range(generations):
            # Evaluate EI for current population
            ei_values = self._acquisition_function(population)
            # Select top 50% as parents
            top_idx = np.argsort(ei_values)[::-1][:pop_size // 2]
            parents = population[top_idx]
            # Generate new population via crossover and mutation
            new_population = []
            for _ in range(pop_size):
                parent1, parent2 = self._select_parents(parents)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, bounds)
                new_population.append(child)
            population = np.array(new_population)
        
        # Select best candidate
        ei_values = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        return population[best_idx].reshape(1, -1)

    def _select_parents(self, parents):
        idx1 = self.rng.integers(len(parents))
        idx2 = self.rng.integers(len(parents))
        return parents[idx1], parents[idx2]

    def _crossover(self, parent1, parent2):
        mask = self.rng.integers(0, 2, self.dim).astype(bool)
        return np.where(mask, parent1, parent2)

    def _mutate(self, child, bounds):
        mutation_rate = 0.1
        noise = self.rng.normal(0, 0.1, self.dim)
        child += mutation_rate * noise
        return np.clip(child, bounds[0], bounds[1])

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
        best_idx = np.argmin(self.y)
        self.best_y = self.y[best_idx, 0]
        self.best_x = self.X[best_idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial Latin Hypercube design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)

        # Initialize GP model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
