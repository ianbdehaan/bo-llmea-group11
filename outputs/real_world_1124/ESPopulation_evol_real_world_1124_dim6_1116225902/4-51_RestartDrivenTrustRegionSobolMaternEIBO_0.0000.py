 
from collections.abc import Callable  
from scipy.stats import qmc, norm  
import numpy as np  
from sklearn.gaussian_process import GaussianProcessRegressor  
from sklearn.gaussian_process.kernels import Matern  

class RestartDrivenTrustRegionSobolMaternEIBO:  
    def __init__(self, budget: int, dim: int):  
        self.budget = budget  
        self.dim = dim  
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])  
        self.X: np.ndarray = None  
        self.y: np.ndarray = None  
        self.n_evals = 0  
        self.n_init = 2 * dim + 1  # Expanded initial design  
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))  
        self.gp = None  
        self.best_point: np.ndarray = None  
        self.trust_radius = 3.0  # Fixed trust region size  
        self.restart_counter = 0  
        self.max_no_progress = 5  # Restart threshold  

    def _sample_points(self, n_points):  
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)  
        samples = sobol.random(n=n_points)  
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples  

    def _fit_model(self, X, y):  
        kernel = 1.0 * Matern(length_scale=np.ones(self.dim), nu=5/2,  
                              length_scale_bounds=(1e-2, 1e2))  
        gp = GaussianProcessRegressor(  
            kernel=kernel,  
            alpha=1e-6,  
            n_restarts_optimizer=1,  # Lightweight refit  
            random_state=int(self.rng.integers(2**31 - 1))  
        )  
        gp.fit(X, y)  
        return gp  

    def _select_next_points(self, batch_size=2):  
        # Generate candidates within trust region  
        current_lower = np.clip(self.best_point - self.trust_radius, self.bounds[0], self.bounds[1])  
        current_upper = np.clip(self.best_point + self.trust_radius, self.bounds[0], self.bounds[1])  
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)  
        candidates = sobol.random(n=100)  
        candidates = current_lower + candidates * (current_upper - current_lower)  

        # Compute EI for batch selection  
        mean, std = self.gp.predict(candidates, return_std=True)  
        mean = mean.flatten()  
        std = std.flatten()  
        current_best = np.min(self.y)  

        mask = std == 0  
        z = (current_best - mean) / std  
        z = np.clip(z, -8, 8)  
        ei = np.zeros_like(mean)  
        ei[~mask] = (current_best - mean[~mask]) * norm.cdf(z[~mask]) + std[~mask] * norm.pdf(z[~mask])  
        ei[mask] = 0.0  

        # Select top batch_size candidates  
        best_idx = np.argpartition(-ei, batch_size)[:, :batch_size]  
        return candidates[best_idx].reshape(-1, self.dim)  

    def _evaluate_points(self, func, X):  
        y = np.array([func(x) for x in X])  
        return y[:, np.newaxis]  

    def _update_eval_points(self, new_X, new_y):  
        if self.X is None:  
            self.X = new_X  
            self.y = new_y  
        else:  
            self.X = np.vstack((self.X, new_X))  
            self.y = np.vstack((self.y, new_y))  

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:  
        # Initial design  
        self.X = self._sample_points(self.n_init)  
        self.y = self._evaluate_points(func, self.X)  
        self.n_evals = self.X.shape[0]  
        self.best_point = self.X[np.argmin(self.y)]  
        self.gp = self._fit_model(self.X, self.y)  
        self.restart_counter = 0  

        while self.n_evals < self.budget:  
            # Select batch of candidates  
            batch_size = 2  
            remaining_budget = self.budget - self.n_evals  
            actual_batch = min(batch_size, remaining_budget)  
            next_points = self._select_next_points(batch_size=actual_batch)  
            new_y = self._evaluate_points(func, next_points)  
            self._update_eval_points(next_points, new_y)  
            self.n_evals += actual_batch  

            # Track progress for restart decision  
            prev_best = np.min(self.y)  
            self.best_point = self.X[np.argmin(self.y)]  
            if prev_best > np.min(self.y):  # Improvement detected  
                self.restart_counter = 0  
            else:  
                self.restart_counter += 1  

            # Restart logic: Re-initialize trust region around best point  
            if self.restart_counter >= self.max_no_progress:  
                self.trust_radius = 3.0  # Reset to full size  
                self.gp = self._fit_model(self.X, self.y)  # Refit model  
                self.restart_counter = 0  

        best_idx = np.argmin(self.y)  
        return self.y[best_idx], self.X[best_idx]  
