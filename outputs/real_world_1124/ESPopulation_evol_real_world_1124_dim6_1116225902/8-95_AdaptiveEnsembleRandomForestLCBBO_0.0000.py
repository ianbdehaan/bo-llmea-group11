from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution

class AdaptiveEnsembleRandomForestLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Maximin Latin Hypercube design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_sampler = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.batch_size = min(2, self.budget // 5)  # Small adaptive batch size
        self.refit_interval = 5  # Model refitting frequency

    def _sample_points(self, n_points):
        """Generate maximin Latin Hypercube samples."""
        samples = self.lhs_sampler.random(n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        """Train a random forest regressor with variance estimation."""
        model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            max_features='sqrt',
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        model.fit(X, y)
        return model

    def _predict_with_var(self, X):
        """Estimate mean and variance using tree-level predictions."""
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = preds.mean(axis=0)
        var = preds.var(axis=0)
        return mean, var

    def _select_next_points(self):
        """Optimize LCB using differential evolution with diversity penalties."""
        def acquisition(x, prev_points=None):
            x = x.reshape(1, -1)
            mean, var = self._predict_with_var(x)
            beta = 2.0 * np.log(self.n_evals + 1)  # Adaptive β schedule
            lcb = mean[0] - beta * np.sqrt(var[0])
            
            # Penalize proximity to previously selected points
            if prev_points is not None and len(prev_points) > 0:
                dist = np.min(np.sum((x - prev_points)**2, axis=1))
                penalty = 100 * (1.0 - np.exp(-dist * 5))
                return lcb + penalty[0]
            return lcb

        lb, ub = self.bounds[0], self.bounds[1]
        bounds = [(l, u) for l, u in zip(lb, ub)]
        selected_points = np.zeros((self.batch_size, self.dim))
        
        for i in range(self.batch_size):
            result = differential_evolution(
                lambda x: acquisition(x, prev_points=selected_points[:i] if i > 0 else None),
                bounds,
                strategy='best1bin',
                maxiter=10,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.8,
                seed=int(self.rng.integers(2**31 - 1))
            )
            selected_points[i] = result.x
        return selected_points

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
            self.best_X = new_X[np.argmin(new_y)].copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx] < self.y[np.argmin(self.y)]:
                self.best_X = self.X[current_best_idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main optimization loop
        iteration = 0
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)
                
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
