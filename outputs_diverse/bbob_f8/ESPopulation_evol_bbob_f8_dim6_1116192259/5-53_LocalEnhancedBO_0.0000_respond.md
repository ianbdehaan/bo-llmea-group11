# Description
**LocalEnhancedBO**: This Bayesian Optimization algorithm employs a local enhancement strategy that combines multiple surrogate models with adaptive acquisition functions. The key innovation is using a local Gaussian Process model around the current best point, complemented by a global model for exploration. It dynamically switches between expected improvement and lower confidence bound acquisition functions based on the optimization stage. The algorithm uses a hybrid sampling approach for initial design and employs efficient local search strategies for acquisition function optimization.

# Justification
The proposed LocalEnhancedBO algorithm differs significantly from existing approaches in several key aspects:

1. **Dual Surrogate Modeling**: Unlike single-model approaches, it maintains both local and global Gaussian Process models, allowing for more nuanced exploration-exploitation trade-offs.

2. **Adaptive Acquisition Functions**: Instead of fixed acquisition functions, it dynamically switches between EI and LCB based on optimization stage, adapting to both early exploration and late exploitation phases.

3. **Local Enhancement Strategy**: The algorithm focuses computational resources on a local region around the best point for fine-tuning, while maintaining global coverage through the global model.

4. **Hybrid Sampling**: Uses Latin hypercube sampling for initial design but dynamically adjusts based on performance feedback.

5. **Efficient Optimization**: Employs a combination of random sampling and local refinement for acquisition function optimization to balance computational cost with quality.

This approach is expected to be more efficient than existing methods because it focuses computation where it's most beneficial (around the best point) while maintaining global search capabilities, and adapts its strategy based on problem characteristics rather than using fixed heuristics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
import warnings

class LocalEnhancedBO:
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
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Surrogate model parameters
        self.global_gp = None
        self.local_gp = None
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.lcb_weight = 0.1  # Weight for LCB acquisition function
        
        # Local model parameters
        self.local_radius_factor = 0.5
        self.min_local_radius = 0.1
        self.max_local_radius = 2.0
        
        # Adaptive acquisition switching
        self.switch_threshold = 0.3  # Switch from LCB to EI after this fraction of budget
        self.acquisition_type = "lcb"  # Start with LCB for exploration
        
        # Performance tracking
        self.best_y_history = []
        self.iteration = 0

    def _sample_points(self, n_points):
        """Sample points using Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit Gaussian Process surrogate model"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _local_model_radius(self):
        """Calculate local model radius based on current best"""
        if self.X is None or len(self.X) == 0:
            return self.min_local_radius
            
        # Use the distance to the farthest point as a reference
        best_x = self.X[np.argmin(self.y)]
        
        # Calculate distances from best point to all points
        distances = np.linalg.norm(self.X - best_x, axis=1)
        
        # Use median distance as radius (scaled by factor)
        radius = np.median(distances) * self.local_radius_factor
        
        # Clip to bounds
        radius = np.clip(radius, self.min_local_radius, self.max_local_radius)
        
        return radius

    def _acquisition_function(self, X):
        """Adaptive acquisition function switching between EI and LCB"""
        if self.global_gp is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.global_gp.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        if self.acquisition_type == "ei":
            # Expected Improvement
            improvement = y_min - mu
            z = improvement / sigma
            acq = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        else:
            # Lower Confidence Bound
            acq = mu - self.lcb_weight * sigma
            
        return acq.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select next points using adaptive acquisition function"""
        # Use a reasonable number of starting points for optimization
        n_starting_points = min(max(50 * self.dim, 100), 500)
        
        if self.X is not None and len(self.X) > 0:
            # Sample around the current best point for local refinement
            best_x = self.X[np.argmin(self.y)]
            
            # Generate candidates in a local region around the best point
            local_radius = self._local_model_radius()
            candidates = []
            
            while len(candidates) < n_starting_points:
                # Generate random points within local radius
                sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * local_radius
                
                # Ensure points are within bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                candidates.append(sample)
            
            candidates = np.array(candidates)
        else:
            # If no points yet, use Latin hypercube sampling
            candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates (use a simple heuristic - just the best ones)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        return selected_points

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

    def _update_acquisition_strategy(self):
        """Update acquisition function based on optimization progress"""
        # Switch to EI after a certain fraction of budget is used
        if self.iteration > 0 and self.n_evals / self.budget > self.switch_threshold:
            self.acquisition_type = "ei"

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial global GP model
        self.global_gp = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self.global_gp = self._fit_gp_model(self.X, self.y)
            
            # Update acquisition strategy
            self._update_acquisition_strategy()
            
            # Select next points using acquisition function
            next_points = self._select_next_points(1)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            self.iteration += 1
            
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
  File "<LocalEnhancedBO>", line 207, in __call__
 207->             next_points = self._select_next_points(1)
  File "<LocalEnhancedBO>", line 148, in _select_next_points
 148->         acq_values = self._acquisition_function(candidates)
  File "<LocalEnhancedBO>", line 114, in _acquisition_function
 112 |             improvement = y_min - mu
 113 |             z = improvement / sigma
 114->             acq = improvement * norm.cdf(z) + sigma * norm.pdf(z)
 115 |         else:
 116 |             # Lower Confidence Bound
NameError: name 'norm' is not defined
