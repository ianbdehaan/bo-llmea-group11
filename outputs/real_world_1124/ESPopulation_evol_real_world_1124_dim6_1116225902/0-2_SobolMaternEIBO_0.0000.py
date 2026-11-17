import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from collections.abc import Callable

class SobolMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(10, 2 * dim)  # Ensure sufficient initial samples
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))  # Seeded RNG

    def _sample_points(self, n_points):
        # Generate quasi-random Sobol samples
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return scaled_samples

    def _fit_model(self, X, y):
        # GP with Matern kernel (ARD)
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement for minimization
        mu, sigma = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mu
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + sigma * norm.pdf(z)
        ei[sigma == 0.] = max(best_y - mu[sigma == 0.], 0.0)
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Optimize EI using random sampling and gradient-based refinement
        n_candidates = 50
        candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(candidates)
        idx = np.argsort(acq_values, axis=0)[::-1]
        top_candidates = candidates[idx[:10]]  # Top 10 candidates for L-BFGS-B
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
        best_x, best_acq = max(res, key=lambda x: x[1])
        return best_x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate func for each row in X
        y = np.array([func(x) for x in X])
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with Sobol samples
        self.X = self._sample_points(self.n_init)
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
            current_y = new_y[0][0]
            if current_y < best_y:
                best_y = current_y
                best_x = next_point[0]

        return best_y, best_x
