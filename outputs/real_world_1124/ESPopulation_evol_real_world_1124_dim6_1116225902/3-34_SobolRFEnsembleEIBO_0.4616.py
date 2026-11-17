from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import warnings

class SobolRFEnsembleEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Larger initial design for ensemble training
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))

    def _sample_points(self, n_points):
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        model = ExtraTreesRegressor(
            n_estimators=50,
            max_depth=5,
            max_features="sqrt",
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Ensure single-core usage
        )
        model.fit(X, y.ravel())
        return model

    def _acquisition_function(self, X):
        # Collect predictions from all trees in the ensemble
        n_trees = len(self.model.estimators_)
        all_predictions = np.zeros((X.shape[0], n_trees))
        for i, tree in enumerate(self.model.estimators_):
            all_predictions[:, i] = tree.predict(X)
        mean = np.mean(all_predictions, axis=1)
        std = np.std(all_predictions, axis=1)
        best_y = np.min(self.y)
        improvement = best_y - mean  # For minimization
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle zero variance cases
        return ei

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(50)  # Generate 50 Sobol candidates
        acq_vals = self._acquisition_function(candidates)
        best_acq_idx = np.argmax(acq_vals)
        next_X = candidates[best_acq_idx].reshape(1, -1)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings for small data
            self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points(1)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model with updated data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
