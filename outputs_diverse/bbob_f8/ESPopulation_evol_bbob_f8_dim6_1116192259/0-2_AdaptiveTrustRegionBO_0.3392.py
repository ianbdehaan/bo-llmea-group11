from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np

class AdaptiveTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10)  # Initial points - start with more points for higher dimensions

        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.tr_radius = 1.0  # Initial trust region radius
        self.tr_center = None  # Center of current trust region
        
        # Convergence tracking
        self.best_y_history = []
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 5

    def _sample_points(self, n_points):
        # Sample points using Latin hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Using Gaussian Process with ARD Matern kernel
        # We use a fixed length scale for efficiency and avoid overfitting
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,  # Limited restarts for computational efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, best_y):
        # Implement expected improvement acquisition function
        # EI = (mu(x) - y_best) * Phi((mu(x) - y_best) / sigma(x)) + sigma(x) * phi((mu(x) - y_best) / sigma(x))
        
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)  # Avoid numerical issues
        
        # Calculate EI
        improvement = mu - best_y
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # 1. Sample random points within trust region
        # 2. Use L-BFGS-B optimization on acquisition function
        # 3. Select best points from both approaches
        
        n_random = max(10, batch_size // 2)  # Sample some random points
        n_optimize = batch_size - n_random
        
        # Sample random points within trust region
        if self.tr_center is not None:
            # Create bounds within trust region
            local_bounds = []
            for i in range(self.dim):
                lower = max(self.bounds[0][i], self.tr_center[i] - self.tr_radius)
                upper = min(self.bounds[1][i], self.tr_center[i] + self.tr_radius)
                local_bounds.append((lower, upper))
            
            # Generate random samples within trust region
            random_points = np.zeros((n_random, self.dim))
            for i in range(n_random):
                for j in range(self.dim):
                    random_points[i, j] = self.rng.uniform(local_bounds[j][0], local_bounds[j][1])
        else:
            # If no trust region defined yet, use full bounds
            random_points = self._sample_points(n_random)
        
        # Optimize acquisition function using L-BFGS-B from multiple starting points
        if self.X is not None and len(self.X) > 0:
            best_y = np.min(self.y)
            model = self._fit_model(self.X, self.y)
            
            # Generate starting points for optimization
            n_starts = min(5, max(1, n_optimize // 2))
            if self.tr_center is not None:
                start_points = np.zeros((n_starts, self.dim))
                for i in range(n_starts):
                    for j in range(self.dim):
                        start_points[i, j] = self.rng.uniform(
                            max(self.bounds[0][j], self.tr_center[j] - self.tr_radius),
                            min(self.bounds[1][j], self.tr_center[j] + self.tr_radius)
                        )
            else:
                start_points = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_opt_point = None
            
            for start_point in start_points:
                def obj(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x, model, best_y)[0, 0]
                
                try:
                    res = minimize(obj, start_point, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
                    if -res.fun > best_acq_val:
                        best_acq_val = -res.fun
                        best_opt_point = res.x
                except:
                    continue
            
            if best_opt_point is not None:
                optimize_points = best_opt_point.reshape(1, -1)
            else:
                optimize_points = self._sample_points(1)
        else:
            optimize_points = self._sample_points(1)
        
        # Combine points
        if n_optimize > 0 and len(optimize_points) > 0:
            selected_points = np.vstack([random_points, optimize_points[:n_optimize]])
        else:
            selected_points = random_points
        
        # Limit to batch_size
        if len(selected_points) > batch_size:
            selected_points = selected_points[:batch_size]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_values = []
        for point in X:
            y_values.append(func(point))
        
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        
        # Set initial trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Check if we should restart or shrink trust region due to no improvement
            if len(self.best_y_history) > 0:
                current_best = np.min(self.y)
                self.best_y_history.append(current_best)
                
                if current_best >= self.best_y_history[-2] - 1e-4:  # No significant improvement
                    self.consecutive_no_improvement += 1
                    if self.consecutive_no_improvement >= self.max_no_improvement:
                        # Restart with new random points
                        self.consecutive_no_improvement = 0
                        # Reduce trust region radius
                        self.tr_radius = max(self.tr_radius * 0.5, 0.1)
                else:
                    self.consecutive_no_improvement = 0
            else:
                self.best_y_history.append(np.min(self.y))
            
            # Select next points to evaluate
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size to remaining budget
            if batch_size <= 0:
                break
                
            next_points = self._select_next_points(batch_size)
            
            # Evaluate new points
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            
            # Update trust region center to the best point so far
            best_idx = np.argmin(self.y)
            self.tr_center = self.X[best_idx].copy()
            
            # Possibly increase trust region radius if progress is good
            if len(self.best_y_history) > 1 and self.best_y_history[-1] < self.best_y_history[-2]:
                self.tr_radius = min(self.tr_radius * 1.1, 5.0)  # Increase trust region but cap it
            
            # Early termination for small budgets to avoid excessive overhead
            if self.n_evals >= self.budget - 1:
                break
                
        # Return the best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
