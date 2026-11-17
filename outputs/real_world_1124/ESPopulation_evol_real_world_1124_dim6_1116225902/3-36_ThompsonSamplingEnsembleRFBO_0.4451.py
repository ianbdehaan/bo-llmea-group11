from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import math

class ThompsonSamplingEnsembleRFBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Expanded initial design
        self.refit_interval = 5  # Model refits every 5 iterations
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.scaler = StandardScaler()

    def _sample_points(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Normalize inputs for better tree performance
        X_scaled = self.scaler.fit_transform(X)
        
        model = ExtraTreesRegressor(
            n_estimators=20,
            max_features='sqrt',
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Ensures single-core execution
        )
        model.fit(X_scaled, y.ravel())
        return model

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(200)  # 200 Sobol candidates for TS
        X_scaled = self.scaler.transform(candidates)
        
        # Get mean and variance from the ensemble
        means = []
        vars = []
        for tree in self.model.estimators_:
            pred = tree.predict(X_scaled)
            means.append(np.mean(pred, axis=0))
            vars.append(np.var(pred, axis=0))
        
        # Use ensemble mean and variance for sampling
        mean = np.mean(means, axis=0)
        variance = np.mean(vars, axis=0)
        
        # Sample synthetic function values
        rng = np.random.default_rng(int(self.rng.integers(2**31 - 1)))
        synthetic_values = mean + rng.normal(0, np.sqrt(variance))
        
        best_idx = np.argmin(synthetic_values)
        next_X = candidates[best_idx].reshape(1, -1)
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
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with batch processing
        iteration = 0
        while self.n_evals < self.budget:
            if iteration % self.refit_interval == 0:
                self.model = self._fit_model(self.X, self.y)  # Refit periodically
                self.scaler = StandardScaler().fit(self.X)  # Update input normalization
                
            batch_size = 2  # Evaluate two points in parallel
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
