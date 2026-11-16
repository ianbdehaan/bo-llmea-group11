from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
import warnings

class EfficientBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, dim * 3))  # Adaptive initial points based on dimensionality
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model
        self.model = None
        
        # Acquisition function parameters
        self.noise_var = 1e-6  # Small noise variance for numerical stability

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a Matern kernel with automatic relevance detection (ARD)
        # Add WhiteKernel for noise modeling
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(
            noise_level=self.noise_var, noise_level_bounds=(1e-10, 1e+1)
        )
        
        # Use a sparse GP for computational efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # No additional regularization
            normalize_y=True,
            n_restarts_optimizer=2,  # Few restarts to keep cost low
            random_state=int(self.rng.integers(2**31 - 1)),
            noise="gaussian"
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI) acquisition function
        # This is a computationally efficient implementation
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Get predictions from the model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Get the best observed value so far
        y_best = np.min(self.y) if len(self.y) > 0 else 0.0
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Compute EI
        improvement = y_best - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using multi-start optimization
        
        # Start with some random points for exploration
        n_random_starts = max(5, min(20, batch_size * 3))
        
        # Generate multiple random starting points
        random_starts = self._sample_points(n_random_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Try different optimization strategies with multiple starting points
        for i in range(n_random_starts):
            start_point = random_starts[i:i+1]
            
            # Try L-BFGS with the current start point
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point.flatten(),
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    candidate = result.x.reshape(1, -1)
                    value = self._acquisition_function(candidate)[0, 0]
                    if value > best_value:
                        best_value = value
                        best_point = candidate
            except:
                pass
                
            # Also try Nelder-Mead for robustness
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 20}
                )
                
                if result.success:
                    candidate = result.x.reshape(1, -1)
                    value = self._acquisition_function(candidate)[0, 0]
                    if value > best_value:
                        best_value = value
                        best_point = candidate
            except:
                pass
        
        # If we didn't find a good point with optimization, return a random one
        if best_point is None:
            best_point = self._sample_points(1)
            
        return np.tile(best_point, (batch_size, 1))

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception:
                # If evaluation fails, use a large value to penalize
                y_vals.append(1e6)
                
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)
    
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
        
        # Initialize with some random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Determine how many points to evaluate in this batch
            remaining_budget = self.budget - self.n_evals
            
            # For the last few evaluations, reduce batch size
            if remaining_budget <= 10:
                batch_size = 1
            elif remaining_budget <= 50:
                batch_size = min(5, remaining_budget)
            else:
                # Use a reasonable batch size (1-3 for efficiency)
                batch_size = min(max(1, int(remaining_budget / 100)), 3)
            
            # If we can't evaluate anything, break
            if batch_size <= 0 or self.n_evals >= self.budget:
                break
                
            # Select next points using acquisition function
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the model with new data
            self._update_eval_points(next_points, next_values)
            
            # Refit the model occasionally (every 5 iterations or so)
            if self.n_evals % 5 == 0:
                try:
                    self.model = self._fit_model(self.X, self.y)
                except Exception as e:
                    # If refitting fails, continue with old model
                    warnings.warn(f"Model refitting failed: {e}")

        # Return the best point found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
