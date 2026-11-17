from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_random_state

class TreeSamplingRandomForestBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * self.dim  # Latin Hypercube initial design
        self.refit_interval = 5  # Refit every 5 iterations
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.n_estimators = 50  # Number of trees in Random Forest

    def _sample_points(self, n_points, method='sobol'):
        if method == 'sobol':
            samples = self.sobol_gen.random(n_points)
        else:  # Latin Hypercube for initial design
            samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features='sqrt',
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single-core training to avoid parallel overhead
        )
        model.fit(X, y.ravel())  # Flatten y for scikit-learn
        return model

    def _select_next_points(self, batch_size=2):
        candidates = self._sample_points(200, method='sobol')  # 200 Sobol candidates
        # Randomly sample a tree from the forest
        tree_idx = self.rng.integers(len(self.model.estimators_))
        sampled_tree = self.model.estimators_[tree_idx]
        # Predict and select the top candidates
        predictions = sampled_tree.predict(candidates)
        best_indices = np.argsort(predictions)[:batch_size]
        next_X = candidates[best_indices]
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Latin Hypercube
        X_init = self._sample_points(self.n_init, method='lhs')
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)  # Refit periodically

            batch_size = 2  # Evaluate two points in parallel
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
