from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import warnings

class DifferentialEvolutionLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = dim + 1  # Initial design size
        self.rng = np.random.default_rng(42)  # Fixed seed for deterministic runs

    def _sample_points(self, n_points):
        """Generate Sobol samples for initial design"""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        return samples * 10.0 - 5.0  # Scale from [0,1]^d to [-5,5]^d

    def _fit_model(self, X, y):
        """Fit a GP with Matérn-ARD kernel"""
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore GP warnings during fit
            gp = GaussianProcessRegressor(
                kernel=kernel,
                optimizer='fmin_l_bfgs_b',
                n_restarts_optimizer=0,  # Minimal restarts for speed
                alpha=1e-6,  # Small noise variance
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        """Decaying LCB acquisition function"""
        mu, sigma = self.model.predict(X, return_std=True)
        t = self.n_evals - self.n_init  # Iteration counter after initial samples
        if t <= 0:
            # Early exploration phase
            kappa = 2.0
        else:
            # Decay exploration over time
            kappa = np.log(1 + t) / t
        return mu - kappa * sigma

    def _select_next_points(self, batch_size):
        """Optimize acquisition function with differential evolution"""
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(
            func=lambda x: -self._acquisition_function(x.reshape(1, -1)),
            bounds=bounds,
            strategy='best1bin',
            maxiter=10,  # Balance between exploration and runtime
            popsize=15,  # Small population for efficiency
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(self.rng.integers(2**31 - 1)),  # Ensure reproducibility
            polish=True,  # Final L-BFGS-B polish of the best candidate
            init='sobol'  # Space-filling initialization
        )
        return np.array([result.x])

    def _evaluate_points(self, func, X):
        """Evaluate points with function"""
        n_points = len(X)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update history of evaluated points"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """Main optimization loop"""
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            # Select next point to evaluate
            next_point = self._select_next_points(1)
            # Evaluate next point
            next_y = self._evaluate_points(func, next_point)
            # Update history
            self._update_eval_points(next_point, next_y)
            # Update model
            self.model = self._fit_model(self.X, self.y)
            # Update best
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx] < best_y:
                best_y = self.y[current_best_idx]
                best_x = self.X[current_best_idx]

        return best_y, best_x
