from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TrustRegionBatchRestartEnsembleThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = self.dim + 1  # Minimal initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

        # Trust regions: list of dicts with bounds, best_x, best_y, stagnation_count
        self.trust_regions = []
        self.max_regions = 3  # Maximum number of active trust regions
        self.restart_threshold = 5  # Stagnation steps before replacement

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        sigma = np.clip(sigma, 1e-9, None)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        all_candidates = []
        all_acq_values = []

        for region in self.trust_regions:
            candidates = self._sample_points(
                batch_size, bounds=region["bounds"]
            )
            acq_values = self._acquisition_function(candidates)
            all_candidates.append(candidates)
            all_acq_values.append(acq_values)

        all_candidates = np.vstack(all_candidates)
        all_acq_values = np.vstack(all_acq_values)

        # Select top candidates from all regions
        idxs = np.argsort(all_acq_values, axis=0)[:, 0]
        top_candidates = all_candidates[idxs[-5:]]
        optimized_points = []

        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=self.bounds.T,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return all_candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
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

    def _update_regions(self, new_x, new_y):
        best_region_idx = None
        best_region_improvement = False

        for idx, region in enumerate(self.trust_regions):
            if region["bounds"][0].all() <= new_x <= region["bounds"][1].all():
                # Update best in this region
                if new_y < region["best_y"]:
                    region["best_x"] = new_x.copy()
                    region["best_y"] = new_y
                    region["stagnation_count"] = 0
                    best_region_idx = idx
                    best_region_improvement = True
                else:
                    region["stagnation_count"] += 1
            else:
                region["stagnation_count"] += 1

        # Replace stagnant regions
        if best_region_idx is not None:
            for idx, region in enumerate(self.trust_regions):
                if idx != best_region_idx and region["stagnation_count"] >= self.restart_threshold:
                    self.trust_regions.pop(idx)
                    break

        # Add new region if under capacity
        if len(self.trust_regions) < self.max_regions:
            new_bounds = np.array([
                self.rng.uniform(-5.0, 0.0, self.dim),
                self.rng.uniform(0.0, 5.0, self.dim)
            ])
            self.trust_regions.append({
                "bounds": np.clip(new_bounds, -5.0, 5.0),
                "best_x": None,
                "best_y": float("inf"),
                "stagnation_count": 0
            })

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with minimal Latin Hypercube design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize trust regions
        self.trust_regions = [{
            "bounds": np.array([
                np.clip(best_x - 1.0, -5.0, 5.0),
                np.clip(best_x + 1.0, -5.0, 5.0)
            ]),
            "best_x": best_x,
            "best_y": best_y,
            "stagnation_count": 0
        }]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update global best
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            # Update trust regions
            self._update_regions(next_point[0], new_y[0])

            # Refit model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
