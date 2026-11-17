from collections.abc import Callable
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import qmc
import time

class GradientForestHybridUCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        
        # Parameters
        self.n_init = max(2 * self.dim + 1, 20)  # Latin hypercube initial design
        self.batch_size = min(3, self.budget // 20)  # Adaptive batch size
        self.rf_estimators = 50  # Number of trees in random forest
        self.ucb_beta = 0.2  # UCB exploration parameter
        self.novelty_weight = 0.4  # Weight for the novelty component in the acquisition function
        self.refit_interval = 3  # Base refit frequency
        
        # Random state
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_design = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))

    def _sample_points(self, n_points, trust_center=None, radius=None):
        """Sample points with Latin Hypercube Sampling"""
        if trust_center is None:
            # Global sampling
            samples = self.lhs_design.random(n_points)
        else:
            # Local sampling around trust_center with adaptive radius
            trust_center = np.clip(trust_center, -5.0, 5.0)
            samples = trust_center + self.rng.uniform(-radius, radius, size=(n_points, self.dim))
            samples = np.clip(samples, -5.0, 5.0)
        return samples

    def _fit_model(self, X, y):
        """Fit a Random Forest model with quantile regression for uncertainty estimation"""
        # Split data for better generalization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=int(self.rng.integers(2**31 - 1))
        )
        
        # Create quantile regression model
        rf = RandomForestRegressor(
            n_estimators=self.rf_estimators,
            criterion='squared_error',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1  # Parallelize over CPU cores
        )
        rf.fit(X_train, y_train)
        
        return rf

    def _predict(self, model, X):
        """Get mean prediction and uncertainty bounds"""
        means = model.predict(X)
        # Estimate uncertainty using quantiles from the tree predictions
        preds = np.stack([tree.predict(X) for tree in model.estimators_])
        lower = np.percentile(preds, 5, axis=0)
        upper = np.percentile(preds, 95, axis=0)
        return means, lower, upper

    def _calculate_novelty(self, candidates):
        """Calculate novelty based on distance to existing data points"""
        if self.X is None or len(self.X) == 0:
            return np.zeros(len(candidates))
            
        distances = np.min(np.linalg.norm(candidates[:, np.newaxis] - self.X, axis=2), axis=1)
        max_distance = np.max(distances)
        if max_distance == 0:
            return np.zeros(len(candidates))
        return distances / max_distance

    def _select_next_points(self):
        """Select next batch of points using hybrid UCB + novelty acquisition function"""
        # Generate candidates around the best point
        n_candidates = 100  # Number of candidates to generate per iteration
        candidates = self._sample_points(n_candidates, self.best_X, self.region_radius)
        
        # Predict with the model
        means, lower, upper = self._predict(self.model, candidates)
        uncertainty = upper - lower
        
        # Calculate novelty for each candidate
        novelty = self._calculate_novelty(candidates)
        
        # Hybrid acquisition function
        acq_values = means + self.ucb_beta * uncertainty + self.novelty_weight * novelty
        
        # Select top candidates based on acquisition function
        top_indices = np.argsort(acq_values)  # Lower is better for minimization
        
        # Select batch using clustering to maintain diversity
        if len(candidates) >= self.batch_size:
            top_candidates = candidates[top_indices[:int(len(candidates) * 0.3)]]  # Take top 30%
            kmeans = KMeans(n_clusters=self.batch_size, n_init=1, random_state=int(self.rng.integers(2**31 - 1)))
            kmeans.fit(top_candidates)
            next_X = kmeans.cluster_centers_
        else:
            next_X = candidates[top_indices[:self.batch_size]]
        
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate points and update evaluation count"""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update the evaluation points and adjust the trust region"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
            self.best_X = new_X[np.argmin(new_y)].copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            if np.min(new_y) < np.min(self.y):
                self.best_X = new_X[np.argmin(new_y)].copy()
                # Expand trust region when progress is made
                self.region_radius = min(self.region_radius * 1.2, 5.0)
            else:
                # Contract if no progress
                self.region_radius = max(self.region_radius * 0.8, 0.5)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        self.region_radius = 1.0  # Initial trust region radius
        self.best_X = self.X[np.argmin(self.y)].copy()

        iteration = 0
        while self.n_evals < self.budget:
            # Refit model conditionally based on trust region progress
            if iteration % self.refit_interval == 0 or (self.region_radius >= 1.5 and iteration % 2 == 0):
                self.model = self._fit_model(self.X, self.y)
            
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
