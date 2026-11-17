from collections.abc import Callable
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DifferentialEvolutionThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        # Latin Hypercube for initialization
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,  # Reduced for efficiency
            optimizer='fmin_l_bfgs_b',
            alpha=1e-6,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1)

    def _select_next_points(self, batch_size):
        # Define bounds for DE
        lower, upper = self.bounds[0], self.bounds[1]

        def objective(x):
            x = np.atleast_2d(x)
            return -self._acquisition_function(x)

        # Run DE to optimize acquisition function
        result = differential_evolution(
            func=objective,
            bounds=[(l, u) for l, u in zip(lower, upper)],
            strategy='best1bin',
            maxiter=3,  # Small number of generations for efficiency
            popsize=10,  # Small population size
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=self.rng,
            workers=-1  # Enable parallelization
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
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
