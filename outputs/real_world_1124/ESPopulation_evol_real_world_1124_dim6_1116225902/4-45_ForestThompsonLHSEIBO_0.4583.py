from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class ForestThompsonLHSEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 3 * dim)  # More points for forest initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_sampler = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        # Forest parameters
        self.n_trees = 10  # Small ensemble for speed
        self.batch_size = 2  # Parallel evaluations per iteration
        self.candidates_per_iter = 50  # Diverse candidates for Thompson Sampling

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Latin Hypercube candidates for initial design."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.lhs_sampler.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Random Forest with uncertainty estimation."""
        # Use pipeline with standardization for stability
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('forest', RandomForestRegressor(
                n_estimators=self.n_trees,
                random_state=int(self.rng.integers(2**31 - 1))
            ))
        ])
        pipeline.fit(X, y.ravel())
        return pipeline

    def _acquisition_function(self, model, X, n_samples=1):
        """Thompson Sampling: Sample n function realizations and select minima."""
        # Get predictions from all trees in the forest for each point
        tree_preds = np.array([tree.predict(X) for tree in model.named_steps['forest'].estimators_])
        means = tree_preds.mean(axis=0)
        stds = tree_preds.std(axis=0)
        # Sample from predictive distribution
        samples = means + stds * self.rng.standard_normal(size=(n_samples, len(X)))
        return samples

    def _select_next_points(self):
        """Generate candidates and select diverse top batch via Thompson Sampling."""
        # Generate candidates covering entire domain
        candidates = self._sample_points(self.candidates_per_iter)
        # Sample multiple function realizations
        samples = self._acquisition_function(self.model, candidates, n_samples=5)
        # Find minima for each sample and select top candidates
        min_indices = np.argmin(samples, axis=1)
        unique_indices = np.unique(min_indices)
        selected_indices = unique_indices[:self.batch_size]
        return candidates[selected_indices]

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Latin Hypercube
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Generate and select next batch
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model every 3 iterations to reduce overhead
            if (self.n_evals % 3) == 0:
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
