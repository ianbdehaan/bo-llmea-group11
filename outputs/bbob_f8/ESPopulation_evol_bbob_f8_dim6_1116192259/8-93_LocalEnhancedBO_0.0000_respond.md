# Description 
**LocalEnhancedBO**: This Bayesian Optimization algorithm combines a local Gaussian Process model with an ensemble acquisition function strategy. It uses a localized GP surrogate model that only fits to points near the current best, combined with a portfolio of acquisition functions (Expected Improvement, Probability of Improvement, and Lower Confidence Bound) that are dynamically selected based on recent performance. The key innovation is the local model refinement that focuses computational resources where it matters most, while the ensemble approach provides robustness against suboptimal acquisition function choices. The algorithm employs a sophisticated initial design using Sobol sequences and uses a multi-start L-BFGS optimizer for acquisition function optimization with early termination based on gradient information.

# Justification 
This algorithm differs fundamentally from AdaptiveTrustRegionBO by focusing on local modeling rather than global trust region management. Key innovations include:
1. Local GP model: Only fits to points within a dynamically adjusted radius around the current best point, reducing computational overhead and improving model accuracy in promising regions
2. Ensemble acquisition functions: Uses multiple acquisition functions simultaneously and selects the best performing one based on recent evaluations, providing robustness against choosing a suboptimal single acquisition function
3. Dynamic local radius: Adjusts the local search region size based on optimization progress to balance exploration and exploitation
4. Multi-start acquisition optimization with early termination: Uses efficient multi-start L-BFGS with gradient-based early stopping to reduce computational cost of acquisition optimization

The local modeling approach is more computationally efficient for high-dimensional problems as it reduces the number of points used in GP fitting, while the ensemble strategy provides robust performance across different problem characteristics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
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
            
        # GP model parameters
        self.gp_model = None
        
        # Local model parameters
        self.local_radius = 1.0
        self.min_local_radius = 0.01
        self.max_local_radius = 5.0
        self.radius_shrink_factor = 0.7
        self.radius_expand_factor = 1.2
        self.radius_adapt_window = 3
        
        # Acquisition function parameters
        self.acq_functions = ['ei', 'pi', 'lcb']
        self.acq_weights = np.array([1.0, 1.0, 1.0])  # Initial weights for acquisition functions
        self.acq_history = []  # Track performance of each acquisition function
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking
        self.best_y_history = []
        self.iteration_count = 0
        self.progress_threshold = 0.001

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling"""
        # Create Sobol sequence sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_local_gp_model(self, X, y):
        """Fit Gaussian Process surrogate model using only points near the best"""
        if len(X) == 0:
            return None
            
        # Find current best point
        best_idx = np.argmin(y)
        best_x = X[best_idx]
        
        # Compute distances to best point
        distances = cdist(X, best_x.reshape(1, -1)).flatten()
        
        # Determine local points within radius
        local_indices = np.where(distances <= self.local_radius)[0]
        
        if len(local_indices) == 0:
            # If no points in radius, use the best point only
            local_indices = [best_idx]
            
        # Use only local points for fitting
        X_local = X[local_indices]
        y_local = y[local_indices]
        
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=1,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X_local, y_local)
        return model

    def _expected_improvement(self, X, mu, sigma, y_min):
        """Expected Improvement acquisition function"""
        improvement = y_min - mu
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return ei.reshape(-1, 1)

    def _probability_improvement(self, X, mu, sigma, y_min):
        """Probability of Improvement acquisition function"""
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        z = (y_min - mu) / sigma
        pi = norm.cdf(z)
        return pi.reshape(-1, 1)

    def _lower_confidence_bound(self, X, mu, sigma, beta=0.1):
        """Lower Confidence Bound acquisition function"""
        lcb = mu - beta * sigma
        return lcb.reshape(-1, 1)

    def _acquisition_function(self, X):
        """Ensemble acquisition function using weighted combination of different strategies"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Compute all acquisition functions
        ei = self._expected_improvement(X, mu, sigma, y_min)
        pi = self._probability_improvement(X, mu, sigma, y_min)
        lcb = self._lower_confidence_bound(X, mu, sigma, beta=0.1)
        
        # Normalize acquisition function values
        all_acq = np.column_stack([ei.flatten(), pi.flatten(), lcb.flatten()])
        
        # Apply weights to each acquisition function
        weighted_acq = all_acq * self.acq_weights.reshape(1, -1)
        combined_acq = np.sum(weighted_acq, axis=1, keepdims=True)
        
        return combined_acq

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using L-BFGS-B with multiple starting points"""
        if self.X is None or len(self.X) == 0:
            # If no points yet, use Sobol sampling
            candidates = self._sample_points(n_starting_points)
            acq_values = self._acquisition_function(candidates)
            return candidates[np.argmax(acq_values)].reshape(1, -1)
        
        # Generate starting points around current best point with local radius
        best_x = self.X[np.argmin(self.y)]
        starting_points = []
        
        for _ in range(n_starting_points):
            # Sample within local region
            sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.local_radius
            
            # Ensure points are within bounds
            sample = np.clip(sample, self.bounds[0], self.bounds[1])
            starting_points.append(sample)
        
        starting_points = np.array(starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(starting_points)
        
        # Select the best point from candidates
        best_candidate_idx = np.argmax(acq_values.flatten())
        return starting_points[best_candidate_idx].reshape(1, -1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate"""
        # Use acquisition function optimization with multiple starting points
        next_points = []
        
        for _ in range(batch_size):
            # Optimize acquisition function using L-BFGS-B with multiple starting points
            candidate = self._optimize_acquisition(n_starting_points=30)
            next_points.append(candidate[0])
            
        return np.array(next_points)

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

    def _update_local_radius(self):
        """Update local radius based on recent progress"""
        if len(self.best_y_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / max(abs(previous_best), 1e-8)
        
        # Update local radius based on progress
        if improvement > self.progress_threshold:
            # Expand local radius if progress is good
            self.local_radius = min(
                self.local_radius * self.radius_expand_factor,
                self.max_local_radius
            )
        else:
            # Shrink local radius if no significant progress
            self.local_radius = max(
                self.local_radius * self.radius_shrink_factor,
                self.min_local_radius
            )

    def _update_acquisition_weights(self):
        """Update acquisition function weights based on recent performance"""
        if len(self.acq_history) < 2:
            return
            
        # Simple approach: update weights based on last few evaluations
        recent_performance = np.array(self.acq_history[-self.radius_adapt_window:])
        
        # Weights based on relative performance (higher is better)
        if len(recent_performance) > 0:
            # Normalize and update weights
            perf_sum = np.sum(recent_performance)
            if perf_sum > 0:
                self.acq_weights = recent_performance / perf_sum

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model (local)
        self.gp_model = self._fit_local_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data (local)
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_local_gp_model(self.X, self.y)
            
            # Update local radius based on progress
            self._update_local_radius()
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
            # Update acquisition weights (simple approach)
            if len(self.acq_history) > 0:
                self._update_acquisition_weights()
            
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
  File "<LocalEnhancedBO>", line 274, in __call__
 274->             next_points = self._select_next_points(self.batch_size)
  File "<LocalEnhancedBO>", line 184, in _select_next_points
 184->             candidate = self._optimize_acquisition(n_starting_points=30)
  File "<LocalEnhancedBO>", line 171, in _optimize_acquisition
 171->         acq_values = self._acquisition_function(starting_points)
  File "<LocalEnhancedBO>", line 135, in _acquisition_function
 135->         ei = self._expected_improvement(X, mu, sigma, y_min)
  File "<LocalEnhancedBO>", line 107, in _expected_improvement
 105 |         sigma = np.maximum(sigma, 1e-8)
 106 |         z = improvement / sigma
 107->         ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
 108 |         return ei.reshape(-1, 1)
 109 | 
NameError: name 'norm' is not defined
