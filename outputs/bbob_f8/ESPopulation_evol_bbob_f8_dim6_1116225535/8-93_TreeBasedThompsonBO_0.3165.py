from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TreeBasedThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube for space coverage
        self.rf_estimators = 50  # Random Forest ensemble size
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)  # Latin Hypercube
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=self.rf_estimators,
            random_state=int(self.rng.integers(2**31 - 1)),
            verbose=False,
        )
        model.fit(X, y.ravel())
        return model

    def _acquisition_function(self, X):
        # Sample a random tree from the ensemble and return its prediction
        tree_idx = self.rng.integers(0, len(self.model.estimators_))
        tree = self.model.estimators_[tree_idx]
        return tree.predict(X)

    def _select_next_points(self, batch_size):
        # Use Differential Evolution to minimize the acquisition function
        def objective(x):
            return self._acquisition_function(x.reshape(1, -1))

        bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]

        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=10,  # Limited iterations to maintain efficiency
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            seed=self.rng.integers(2**31 - 1),
        )

        if result.success:
            return result.x.reshape(1, -1)
        else:
            # Fallback to random sampling if DE fails
            return self._sample_points(1)

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

        # Initialize Random Forest model
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
