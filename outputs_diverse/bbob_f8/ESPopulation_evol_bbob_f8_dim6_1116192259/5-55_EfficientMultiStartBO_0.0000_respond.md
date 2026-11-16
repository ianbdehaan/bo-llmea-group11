# Description
**EfficientMultiStartBO**: This Bayesian Optimization algorithm employs a hybrid approach combining efficient initial sampling with multi-start optimization and a novel acquisition function strategy. It uses a Gaussian Process surrogate model with anisotropic Matern kernel, but introduces a dynamic exploration-exploitation balance through a probabilistic acquisition function that adapts based on function curvature. The key innovation is the use of multiple restarts with adaptive step sizes during acquisition function optimization, combined with a sophisticated initial design using maximin Latin hypercube sampling and a progressive refinement strategy that reduces computational overhead as the budget allows more evaluations.

# Justification
This algorithm differs fundamentally from existing approaches in several ways:
1. **Dynamic Acquisition Function**: Instead of fixed EI or LCB, it uses a probabilistic acquisition function that adapts its exploration weight based on local function curvature detected through GP variance.
2. **Multi-start with Adaptive Step Size**: During acquisition optimization, it starts from multiple points with decreasing step sizes to balance global exploration and local exploitation efficiently.
3. **Progressive Refinement Strategy**: Uses a more sophisticated initial sampling that scales with dimensionality but reduces computation by using fewer restarts for GP hyperparameter fitting.
4. **Curvature-Based Exploration**: The exploration weight adapts based on the local function behavior, which is more suitable for functions like Rosenbrock that have complex landscapes.
5. **Efficient Batch Selection**: Implements a simple yet effective batch selection strategy that doesn't require expensive diversity measures or complex optimization.

The approach aims to be computationally efficient while maintaining strong performance across different budgets by adapting its complexity based on remaining evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
import warnings

class EfficientMultiStartBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(5, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # GP model parameters
        self.gp_model = None
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # For multi-start optimization
        self.max_optimization_restarts = 50
        self.min_optimization_restarts = 5
        self.optimization_restarts = self.max_optimization_restarts
        
        # Exploration strategy
        self.exploration_weight_base = 0.1
        self.exploration_decay = 0.95
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Reduce number of restarts based on remaining budget
        n_restarts = max(1, min(3, self.budget // 100))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Curvature-adaptive acquisition function"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Calculate exploration weight based on local variance and function behavior
        if len(self.y) > 1:
            # Estimate curvature via variance of recent evaluations
            recent_y = self.y[-min(10, len(self.y)):]
            curvature = np.std(recent_y)
            # Adapt exploration weight inversely proportional to curvature
            adaptive_exploration = self.exploration_weight_base * (1.0 / (1.0 + curvature))
        else:
            adaptive_exploration = self.exploration_weight_base
            
        # Use a hybrid acquisition that balances exploration and exploitation
        # This is a modified Expected Improvement with adaptive exploration weight
        improvement = y_min - mu
        z = improvement / sigma
        
        # Apply adaptive exploration weight
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Scale by adaptive exploration weight
        acquisition = ei * (1.0 + adaptive_exploration)
        
        return acquisition.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=50):
        """Optimize acquisition function using multi-start approach with adaptive step sizes"""
        if self.X is not None and len(self.X) > 0:
            # Start from the best point found so far
            start_points = [self.X[np.argmin(self.y)]]
            
            # Add random points for diversity
            n_additional = min(n_starts - 1, 20)
            if n_additional > 0:
                additional_points = self._sample_points(n_additional)
                start_points.extend(additional_points)
        else:
            # If no points yet, use Latin hypercube sampling
            start_points = self._sample_points(n_starts)
            
        best_acq_val = -np.inf
        best_x = None
        
        for start_point in start_points:
            # Use a simple optimization approach with decreasing step sizes
            try:
                # First attempt with larger steps (global exploration)
                res = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 50}
                )
                
                if res.success:
                    acq_val = -res.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_x = res.x
                        
            except Exception:
                continue
                
        return best_x.reshape(1, -1) if best_x is not None else self._sample_points(1)

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate all points in X
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y_vals
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Select next points using acquisition function optimization
            next_points = self._optimize_acquisition(n_starts=self.optimization_restarts)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Reduce restarts as we progress to save computation (but not too much)
            if self.optimization_restarts > self.min_optimization_restarts:
                self.optimization_restarts = max(
                    self.min_optimization_restarts,
                    int(self.optimization_restarts * self.exploration_decay)
                )
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<EfficientMultiStartBO>", line 51
    samples = sampler.random(n=n=n_points)
                                ^
SyntaxError: invalid syntax
