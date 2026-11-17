from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

class RandomForestDifferentialLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_sampler = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        
        # Acquisition parameters
        self.kappa_start = 5.0
        self.kappa_end = 0.1
        self.current_kappa = self.kappa_start

    def _sample_points(self, n_points):
        """Generate Latin Hypercube candidates for initialization."""
        samples = self.lhs_sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Random Forest with uncertainty estimation via variance."""
        # Add noise to prevent overfitting (common in Random Forests)
        noise = 1e-6 * np.random.randn(*y.shape)
        y_noisy = y + noise
        model = RandomForestRegressor(
            n_estimators=50,  # Lightweight yet robust
            max_depth=10,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_noisy)
        return model

    def _acquisition_function(self, X):
        """Lower Confidence Bound (LCB) with dynamic kappa."""
        # Predict mean and std using Random Forest
        mean = self.model.predict(X).reshape(-1, 1)
        # Compute variance via out-of-bag (OOB) prediction variance
        _, std = self._predict_with_variance(X)
        lcb = mean - self.current_kappa * std
        return lcb.ravel()

    def _predict_with_variance(self, X):
        """Estimate mean and std using tree predictions."""
        n_trees = self.model.n_estimators
        predictions = np.zeros((len(X), n_trees))
        for i in range(n_trees):
            predictions[:, i] = self.model.estimators_[i].predict(X)
        mean = predictions.mean(axis=1).reshape(-1, 1)
        std = predictions.std(axis=1).reshape(-1, 1)
        return mean, std

    def _optimize_acquisition(self, bounds):
        """Optimize LCB using Differential Evolution."""
        def wrapped_acquisition(x):
            x = np.array(x).reshape(1, -1)
            return -self._acquisition_function(x).item()  # Minimize negative LCB
        
        result = differential_evolution(
            wrapped_acquisition,
            bounds=self.bounds.T,
            strategy='best1bin',
            popsize=10,  # Small population for efficiency
            tol=1e-6,
            maxiter=50,
            recombination=0.9,
            mutation=(0.5, 1.0),
            polish=False
        )
        return result.x.reshape(1, -1)

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
            self.X = new_X
            self.y = new_y
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
        
        # Adaptive exploration parameter schedule
        total_iterations = self.budget - self.n_init
        while self.n_evals < self.budget:
            # Update kappa linearly
            remaining = self.budget - self.n_evals
            progress = (total_iterations - remaining) / total_iterations
            self.current_kappa = self.kappa_start - progress * (self.kappa_start - self.kappa_end)
            
            # Optimize acquisition function with DE
            next_X = self._optimize_acquisition(self.bounds)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
