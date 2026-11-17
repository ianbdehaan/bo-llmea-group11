from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics.pairwise import euclidean_distances

class AdaptiveForestEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # 12 for dim=6
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.tree_growth_rate = 2  # Trees added every 10 evaluations
        self.n_trees_initial = 5

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Dynamic number of trees based on data size
        n_trees = self.n_trees_initial + max(0, len(X) // 10) * self.tree_growth_rate
        model = ExtraTreesRegressor(
            n_estimators=n_trees,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1,  # Parallelize training
            verbose=0
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, f_best):
        # Vectorized EI using ensemble variance
        preds = self.model.predict(X)
        preds = np.column_stack([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(preds, axis=1)
        sigma = np.std(preds, axis=1)
        sigma = np.clip(sigma, 1e-9, None)

        imp = f_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[imp < 0] = 0.0
        return ei

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100)  # Larger global pool
        acq = self._acquisition_function(candidates, self.best_y)
        idxs = np.argsort(acq)[::-1][:50]  # Top 50 for diversity

        # MaxMin filtering to ensure diverse candidates
        if len(idxs) > 1:
            distances = euclidean_distances(candidates[idxs])
            distances[range(len(idxs)), range(len(idxs))] = np.inf
            min_distances = np.min(distances, axis=1)
            top_5_indices = np.argsort(min_distances)[-5:]
            top_candidates = candidates[idxs[top_5_indices]]
        else:
            top_candidates = candidates[idxs]

        # Local refinement
        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1), self.best_y).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[0]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points, self.best_y)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

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
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]

            # Prune data: keep top 10% of all evaluations
            if len(self.y) > self.budget // 10:
                idxs = np.argsort(self.y, axis=0)[: self.budget // 10]
                self.X = self.X[idxs].squeeze()
                self.y = self.y[idxs].squeeze()

            # Re-fit model with dynamic complexity
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
