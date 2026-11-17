from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import warnings

class DecayingLCBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 10)  # Improved initialization size
        self.rng = np.random.default_rng(42)

    def _sample_points(self, n_points):
        """Space-filling Sobol sampling"""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        return samples * 10.0 - 5.0  # Map [0,1]^d to [-5,5]^d

    def _fit_model(self, X, y):
        """Fast GP fitting with Matérn-ARD kernel"""
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel,
                optimizer='fmin_l_bfgs_b',
                n_restarts_optimizer=0,
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        """Dynamic LCB with decaying exploration"""
        mu, sigma = self.model.predict(X, return_std=True)
        t = max(1, self.n_evals - self.n_init)
        kappa = np.log(1 + t) / t
        return mu - kappa * sigma

    def _optimize_acq(self, x0):
        """L-BFGS-B optimization of acquisition function"""
        def neg_acq(x):
            return -self._acquisition_function(x.reshape(1, -1))[0]
        
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(
            fun=neg_acq,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        return result

    def _select_next_point(self):
        """Hybrid random + local optimization of acquisition function"""
        n_rand = 30  # Reduced random starts for speed
        candidates = []
        
        # Random initial candidates
        for _ in range(n_rand):
            x0 = self._sample_points(1).ravel()
            result = self._optimize_acq(x0)
            if result.success:
                candidates.append((result.fun, result.x))
        
        # Local optimization from current best
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            x0 = self.X[best_idx].copy()
            # Perturb to avoid local optima
            x0 += self.rng.normal(0, 0.1, self.dim)
            x0 = np.clip(x0, -5.0, 5.0)
            result = self._optimize_acq(x0)
            if result.success:
                candidates.append((result.fun, result.x))
        
        if not candidates:
            return self._sample_points(1)
            
        # Select candidate with best acquisition value
        candidates = np.array(candidates)
        best_idx = np.argmin(candidates[:, 0])
        return candidates[best_idx, 1].reshape(1, -1)

    def _evaluate_points(self, func, X):
        """Safe function evaluation with budget check"""
        n_points = len(X)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            if self.n_evals >= self.budget:
                break
            y[i, 0] = func(X[i])
            self.n_evals += 1
        return y

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
            next_point = self._select_next_point()
            next_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, next_y)
            self.model = self._fit_model(self.X, self.y)
            
            # Update best solution
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx] < best_y:
                best_y = self.y[current_best_idx]
                best_x = self.X[current_best_idx]

        return best_y, best_x
    
    def _update_eval_points(self, new_X, new_y):
        """Update history of evaluated points"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
