from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TrustRegionForestLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = max(2 * dim + 1, 20)  # Latin Hypercube initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None  # Random Forest surrogate
        self.kappa = 1.5  # Fixed exploration coefficient
        self.best_point = None  # Trust region center
        self.radius = 1.0  # Initial radius
        self.radius_min = 0.1  # Minimum radius
        self.radius_max = 5.0  # Maximum radius
        self.patience = 5  # Evaluations without improvement before contraction
        self.patience_counter = 0
        self.last_best_value = np.inf

    def _sample_points(self, n_points):
        # Latin Hypercube Sampling for space-filling initial design
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Random Forest with uncertainty estimation
        model = RandomForestRegressor(
            n_estimators=10,
            random_state=int(self.rng.integers(2**31 - 1)),
            bootstrap=False,
            n_jobs=-1  # Enable parallel training
        )
        model.fit(X, y.ravel())  # Fit on 1D target
        return model

    def _acquisition_function(self, X):
        # Compute LCB using mean and variance from Random Forest
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(tree_preds, axis=0)
        var = np.var(tree_preds, axis=0)
        lcb = mean - self.kappa * np.sqrt(var)
        return lcb.reshape(-1, 1)

    def _acquisition_objective(self, x):
        # Wrapper for differential evolution (maximize LCB)
        x = np.array(x).reshape(1, -1)
        return -self._acquisition_function(x)[0, 0]  # Negative for minimization

    def _select_next_point(self):
        # Determine trust region bounds around the best point
        if self.best_point is None:
            bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        else:
            lower = np.clip(self.best_point - self.radius, self.bounds[0], self.bounds[1])
            upper = np.clip(self.best_point + self.radius, self.bounds[0], self.bounds[1])
            bounds = [(lower[i], upper[i]) for i in range(self.dim)]
        # Differential Evolution within trust region
        result = differential_evolution(
            self._acquisition_objective,
            bounds,
            strategy='best1bin',
            maxiter=5,
            popsize=10,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.rng
        )
        best_candidate = result.x
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate candidates without exceeding budget
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        # Maintain evaluated data
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
        # Initialize trust region
        best_idx = np.argmin(self.y)
        self.best_point = self.X[best_idx].copy()
        self.last_best_value = self.y[best_idx, 0]

        # Main optimization loop
        while self.n_evals < self.budget:
            new_point = self._select_next_point()
            new_y = self._evaluate_points(func, new_point)
            self._update_eval_points(new_point, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += 1

            # Update trust region
            if new_y[0] < self.last_best_value:
                self.last_best_value = new_y[0]
                self.best_point = new_point.copy()
                self.patience_counter = 0
                self.radius = min(self.radius * 1.2, self.radius_max)  # Expand
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.radius = max(self.radius * 0.8, self.radius_min)  # Contract
                    self.patience_counter = 0

        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()
