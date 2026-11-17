from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
import numpy as np
import math

class ExtraTreesLcbDynamicBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Latin Hypercube with maximin-like spacing
        self.refit_interval = 5  # Refit model every 5 iterations
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.latin_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None

    def _sample_points(self, n_points):
        """Generate Latin Hypercube samples"""
        samples = self.latin_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit an ExtraTreesRegressor with bootstrap for uncertainty estimation"""
        model = ExtraTreesRegressor(
            n_estimators=10,
            bootstrap=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _predict_with_variance(self, model, X):
        """Estimate mean and standard deviation using tree ensemble predictions"""
        predictions = np.array([tree.predict(X) for tree in model.estimators_])
        mu = predictions.mean(axis=0)
        sigma = predictions.std(axis=0)
        return mu, sigma

    def _select_next_points(self, batch_size):
        """Optimize LCB acquisition using differential evolution and cluster for diversity"""
        def acquisition_function(x):
            x = x.reshape(1, -1)
            mu, sigma = self._predict_with_variance(self.model, x)
            kappa = self._get_kappa()
            return mu - kappa * sigma  # Minimize LCB

        bounds = [(lb, ub) for lb, ub in zip(self.bounds[0], self.bounds[1])]
        candidates = []

        # Generate diverse candidates via differential evolution
        for _ in range(batch_size):
            result = differential_evolution(
                acquisition_function, 
                bounds,
                strategy='best1bin',
                maxiter=10,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.8,
                seed=self.rng
            )
            candidates.append(result.x)
        candidates = np.array(candidates)

        # Cluster candidates and select one per cluster for diversity
        kmeans = KMeans(n_clusters=batch_size)
        kmeans.fit(candidates)
        cluster_centers = kmeans.cluster_centers_

        selected_points = []
        for center in cluster_centers:
            distances = np.sum((candidates - center) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            selected_points.append(candidates[closest_idx])
        return np.array(selected_points)

    def _get_kappa(self):
        """Dynamic exploration weight: high early, low later"""
        return max(1.0, (self.budget - self.n_evals) / self.budget * 2.0)

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
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main optimization loop
        iteration = 0
        batch_size = 2  # Parallel evaluation of 2 points
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)  # Refit periodically

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
