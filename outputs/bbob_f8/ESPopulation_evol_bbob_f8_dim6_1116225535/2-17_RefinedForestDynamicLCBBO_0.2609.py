from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RefinedForestDynamicLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Budget-aware initialization
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None
        self.kappa_decay = 0.98  # Faster decay for exploration
        self.initial_kappa = 3.0  # Higher initial exploration

    def _sample_points(self, n_points):
        # Sobol for low-discrepancy sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Lighter Random Forest with bootstrap=False
        model = RandomForestRegressor(
            n_estimators=5,
            random_state=int(self.rng.integers(2**31 - 1)),
            bootstrap=False,
            n_jobs=-1  # Parallel training
        )
        model.fit(X, y.ravel())
        return model

    def _acquisition_function(self, X):
        # LCB with variance from Random Forest
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(tree_preds, axis=0)
        var = np.var(tree_preds, axis=0)
        best_y = np.min(self.y)
        t = self.n_evals
        kappa = self.initial_kappa * (self.kappa_decay ** t)
        lcb = mean - kappa * np.sqrt(var)
        return lcb.reshape(-1, 1)

    def _acquisition_objective(self, x):
        # Maximize LCB (negative for minimization)
        x = np.array(x).reshape(1, -1)
        return -self._acquisition_function(x)[0, 0]

    def _select_next_point(self):
        # Adaptive bounds: shrink around best point as iterations progress
        best_idx = np.argmin(self.y)
        best_point = self.X[best_idx]
        margin = 3.0 * (self.bounds[1] - self.bounds[0]) / (self.budget - self.n_evals + 1)
        adaptive_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(
            self._acquisition_objective,
            adaptive_bounds,
            strategy='best1bin',
            maxiter=3,  # Reduced iterations
            popsize=5,  # Reduced population
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.rng
        )
        best_candidate = result.x
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.n_evals += initial_X.shape[0]
        self.model = self._fit_model(self.X, self.y)

        # Main optimization loop
        while self.n_evals < self.budget:
            new_point = self._select_next_point()
            new_y = self._evaluate_points(func, new_point)
            self._update_eval_points(new_point, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += 1

        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()
