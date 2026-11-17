from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
import warnings

class AdaptiveBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Set initial design size based on dimensionality
        # For low dimensions use more points for better exploration
        if dim <= 3:
            self.n_init = max(5, dim + 1)
        elif dim <= 6:
            self.n_init = max(10, 2 * dim)
        else:
            self.n_init = max(15, 3 * dim)
            
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Parameters for adaptive behavior
        self.exploration_schedule = np.linspace(1.0, 0.1, min(50, budget // 10)) if budget > 100 else [1.0]
        self.current_schedule_idx = 0

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process surrogate model with ARD Matern kernel
        # Use a more robust approach for fitting GP
        try:
            # Try different kernel configurations to avoid numerical issues
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e2), nu=2.5)
            
            # Add WhiteKernel for noise modeling
            kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
            
            # Fit GP with robust hyperparameter optimization
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Small regularization to avoid numerical issues
                normalize_y=True,
                n_restarts_optimizer=3,  # Limit restarts for efficiency
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X, y)
            return gp
        except Exception:
            # Fallback to simpler model if complex fitting fails
            warnings.warn("GP fitting failed, using fallback")
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e2), nu=2.5)
            kernel = kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function with dynamic exploration
        if self.X is None or self.y is None:
            # If no data yet, return random values for initial exploration
            return np.random.rand(X.shape[0], 1)
            
        # Fit model on current data
        model = self._fit_model(self.X, self.y)
        
        # Get current best value
        best_y = np.min(self.y)
        
        # Compute acquisition function
        try:
            # Get predictions from GP
            mean, std = model.predict(X, return_std=True)
            
            # Dynamic exploration factor
            if self.current_schedule_idx < len(self.exploration_schedule):
                exploit_factor = self.exploration_schedule[self.current_schedule_idx]
                self.current_schedule_idx += 1
            else:
                exploit_factor = self.exploration_schedule[-1]
            
            # Expected Improvement calculation
            improvement = best_y - mean
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Apply exploration factor
            ei = ei * exploit_factor
            
            return ei.reshape(-1, 1)
        except Exception:
            # Fallback to random acquisition if calculation fails
            return np.random.rand(X.shape[0], 1)

    def _select_next_points(self, batch_size):
        # Select next points using multi-start optimization of acquisition function
        # This is more robust than simple gradient-based methods
        
        # Generate diverse candidates
        n_candidates = max(50, batch_size * 10)
        candidate_points = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for candidates
        acq_values = self._acquisition_function(candidate_points).flatten()
        
        # Select top candidates based on acquisition values
        top_indices = np.argsort(acq_values)[-batch_size:]
        selected_points = candidate_points[top_indices]
        
        # Refine selection using local optimization around best points
        refined_points = []
        for point in selected_points:
            try:
                # Use differential evolution to optimize acquisition locally
                def objective(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x)[0][0]
                
                # Bound the search within small region around current point
                bounds = [(max(self.bounds[0][i], point[i] - 1.0), 
                          min(self.bounds[1][i], point[i] + 1.0)) 
                         for i in range(len(point))]
                
                result = differential_evolution(objective, bounds, seed=int(self.rng.integers(2**31 - 1)), maxiter=20)
                refined_points.append(result.x)
            except:
                # If local optimization fails, keep the original point
                refined_points.append(point)
        
        return np.array(refined_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
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
        
        # Initial design
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're near the budget, evaluate remaining points
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 1:
                break
                
            # Determine batch size (limit to remaining budget)
            batch_size = min(5, remaining_budget)
            
            # Select next points using acquisition function
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Break if budget exhausted
            if self.n_evals >= self.budget:
                break
        
        # Return best value and point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
