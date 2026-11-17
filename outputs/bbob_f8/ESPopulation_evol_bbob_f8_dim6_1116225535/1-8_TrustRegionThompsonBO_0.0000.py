from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Initial Sobol points
        self.tr_radius = 1.0   # Initial trust region radius
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_local_model(self, X, y):
        """Fit GP only to points within the trust region"""
        best_idx = np.argmin(y)
        best_point = X[best_point]
        
        # Define trust region bounds
        lower = np.clip(best_point - self.tr_radius, self.bounds[0], self.bounds[1])
        upper = np.clip(best_point + self.tr_radius, self.bounds[0], self.bounds[1])
        
        # Filter points within trust region
        in_region = np.all((X >= lower) & (X <= upper), axis=1)
        X_local = X[in_region]
        y_local = y[in_region]
        
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        if len(X_local) > 0:
            model.fit(X_local, y_local)
        return model, lower, upper

    def _acquisition_function(self, X):
        # Thompson Sampling: sample a function from the GP posterior
        return self.model.sample_y(X, n_samples=1).reshape(-1, 1)

    def _select_next_points(self):
        # Generate candidates within trust region
        candidates = self._sample_points(100, 
                                       bounds=[self.current_bounds[0], 
                                               self.current_bounds[1]])
        
        # Optimize sampled function from GP
        acq_values = self._acquisition_function(candidates)
        best_acq_idx = np.argmin(acq_values)  # Minimize the sampled function
        x0 = candidates[best_acq_idx]
        
        # Refine with L-BFGS-B within trust region
        def acq_wrapper(x):
            return self._acquisition_function(x.reshape(1, -1)).item()
        
        result = minimize(
            acq_wrapper, 
            x0, 
            bounds=[(l, u) for l, u in zip(self.current_bounds[0], 
                                          self.current_bounds[1])],
            method='L-BFGS-B'
        )
        
        return result.x.reshape(1, -1)

    def _adjust_trust_region(self, improvement_made):
        if improvement_made:
            self.tr_radius = min(5.0, self.tr_radius * 1.2)  # Expand
        else:
            self.tr_radius = max(0.1, self.tr_radius * 0.8)  # Contract

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation with Sobol design
        self.X = self._sample_points(self.n_init)
        self.y = np.array([func(x) for x in self.X]).reshape(-1, 1)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]
        
        while self.n_evals < self.budget:
            # Fit local model and get trust region bounds
            self.model, lower, upper = self._fit_local_model(self.X, self.y)
            self.current_bounds = (lower, upper)
            
            # Select next point
            next_point = self._select_next_points()
            new_y = np.array([func(next_point[0])]).reshape(-1, 1)
            
            # Update history
            self.X = np.vstack([self.X, next_point])
            self.y = np.vstack([self.y, new_y])
            self.n_evals += 1
            
            # Check if improvement was made
            improvement = new_y[0] < best_y
            if improvement:
                best_y = new_y[0]
                best_x = next_point[0]
            
            # Adjust trust region
            self._adjust_trust_region(improvement)
        
        return best_y, best_x
