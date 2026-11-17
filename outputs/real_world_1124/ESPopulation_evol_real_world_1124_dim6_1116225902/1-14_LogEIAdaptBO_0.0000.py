import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from collections.abc import Callable

class LogEIAdaptBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 2 * dim)  # 12 points for 6D
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Separate samplers for initial design and candidate generation
        self._initial_sampler = qmc.LatinHypercube(d=dim, seed=self.rng)
        self._candidate_sampler = qmc.Sobol(d=dim, scramble=True, seed=self.rng)

    def _initial_sample(self, n_points):
        """Generate Latin Hypercube samples for initial design"""
        samples = self._initial_sampler.random(n_points)
        return self._scale_samples(samples)

    def _candidate_sample(self, n_points):
        """Generate Sobol samples for acquisition function optimization"""
        samples = self._candidate_sampler.random(n_points)
        return self._scale_samples(samples)

    def _scale_samples(self, samples):
        """Scale samples from [0,1]^d to the search space"""
        return self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])

    def _fit_model(self, X, y):
        """Fit adaptive GP with Matérn kernel (ARD)"""
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        """LogEI acquisition function for minimization"""
        mu, sigma = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mu
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / sigma
            logEI = (improvement * norm.cdf(z) + sigma * norm.pdf(z)) / np.maximum(1, sigma)
        
        # Handle sigma == 0 cases safely
        if np.any(sigma == 0.):
            logEI[sigma == 0.] = np.log(np.maximum(best_y - mu[sigma == 0.], 1e-8))
        
        return logEI.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Optimize LogEI using gradient-based refinement from random candidates"""
        n_candidates = 100
        candidates = self._candidate_sample(n_candidates)
        acq_values = self._acquisition_function(candidates)
        
        # Find top candidates for L-BFGS-B
        idx = np.argsort(acq_values, axis=0)
        top_candidates = candidates[idx[:20]]
        
        res = []
        for x0 in top_candidates:
            x0 = x0.reshape(1, -1)
            result = minimize(
                lambda x: -self._acquisition_function(x.reshape(1, -1))[0],
                x0=x0,
                bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            res.append((result.x, -result.fun))
        
        # Return best solution
        best_idx = np.argmax([r[1] for r in res])
        return res[best_idx][0].reshape(1, -1)

    def _evaluate_points(self, func, X):
        """Evaluate function at given points"""
        return np.array([func(x) for x in X]).reshape(-1, 1)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design with Latin Hypercube
        self.X = self._initial_sample(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)
        self.n_evals = len(self.X)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            self.n_evals += 1
            
            if new_y[0][0] < best_y:
                best_y = new_y[0][0]
                best_x = next_point[0]

        return best_y, best_x

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
