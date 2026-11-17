from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestThompsonSamplingLHBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # More diverse initial sampling
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None  # Random Forest surrogate model
        self.best_point: np.ndarray = None

    def _sample_points(self, n_points):
        """Latin Hypercube Sampling for initial design and candidate generation"""
        lhd = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = lhd.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        """Trains a Random Forest with fixed parameters for speed"""
        model = RandomForestRegressor(
            n_estimators=10,  # Small ensemble for fast fitting
            max_depth=5,      # Limits model complexity
            max_features='sqrt',  # Reduces overfitting in high dimensions
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single-core for determinism
        )
        model.fit(X, y)
        return model

    def _select_next_points(self, batch_size=2):
        """Thompson Sampling using a randomly selected tree from the Random Forest"""
        # Generate diverse candidates for acquisition function optimization
        candidates = self._sample_points(30)  # 30 candidates per iteration
        
        # Sample a single tree from the Random Forest posterior
        tree_idx = self.rng.integers(0, len(self.model.estimators_))
        tree_model = self.model.estimators_[tree_idx]
        
        # Evaluate the sampled function (tree) on candidates
        preds = tree_model.predict(candidates)
        
        # Select top candidates for evaluation
        best_indices = np.argsort(preds)[:batch_size]
        return candidates[best_indices]

    def _evaluate_points(self, func, X):
        """Evaluates the black-box function at specified points"""
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        """Updates the set of evaluated points"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """Main optimization loop"""
        # Initial design phase
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]
        self.best_point = self.X[np.argmin(self.y)]

        # Iterative optimization phase
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            next_points = self._select_next_points(batch_size=2)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            self.n_evals += next_points.shape[0]
            self.best_point = self.X[np.argmin(self.y)]

        # Return best solution
        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
