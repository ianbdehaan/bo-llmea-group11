# Description 
**EvolvingSurrogateBO**: This Bayesian Optimization algorithm employs an evolving surrogate model that dynamically switches between different kernel types and hyperparameters based on function behavior, combined with a novel acquisition function that balances exploration and exploitation using a dynamic trade-off parameter. The method uses a Gaussian Process surrogate with automatic kernel selection and adapts its search strategy through periodic reinitialization of the GP model structure and a hybrid acquisition optimization approach involving both global sampling and local refinement.

# Justification 
This algorithm differs fundamentally from AdaptiveTrustRegionBO by abandoning fixed trust regions in favor of an evolving surrogate modeling approach. Key innovations include:
1. Dynamic kernel switching: Instead of using a single Matern kernel, it automatically adapts between RBF and Matern kernels based on function smoothness detected through gradient estimation
2. Evolutionary model reinitialization: The GP model is periodically reinitialized with new hyperparameters to avoid getting stuck in local optima
3. Adaptive acquisition trade-off: Uses a dynamic exploration-exploitation balance that decreases over time, unlike the fixed EI approach of AdaptiveTrustRegionBO
4. Hybrid acquisition optimization: Combines random sampling with diversity-promoting selection and local L-BFGS-B refinement for efficient point selection
5. Multi-kernel GP structure: Uses both RBF and Matern kernels to better capture different characteristics of the function landscape

The approach is designed to be more robust to varying function characteristics while maintaining computational efficiency, especially important for the Rosenbrock function which has a complex landscape with many local minima.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EvolvingSurrogateBO:
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
        self.current_kernel_type = "matern"  # Start with Matern kernel
        
        # Model evolution parameters
        self.model_update_frequency = max(10, dim)  # Update model every N iterations
        self.model_reinit_frequency = max(20, 2 * dim)  # Reinitialize model every N iterations
        self.iteration_count = 0
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 1.0  # Dynamic exploration weight
        self.exploration_decay = 0.95  # How fast exploration decreases
        
        # For diversity in batch selection
        self.batch_size = 1
        self.diversity_factor = 0.2  # Weight for diversity in acquisition optimization
        
        # Performance tracking
        self.best_y_history = []
        self.last_improvement_iter = 0

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling properties"""
        # Create Sobol sequence sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit Gaussian Process surrogate model with adaptive kernel selection"""
        # Select kernel based on function characteristics
        if self.iteration_count > 0 and self.iteration_count % 5 == 0:
            # Estimate smoothness of the function using gradient-like approach
            if len(X) >= 4:
                # Simple gradient estimation using finite differences
                try:
                    # Compute pairwise distances
                    dists = cdist(X, X)
                    # Set diagonal to large value to avoid self-distance
                    np.fill_diagonal(dists, 1e10)
                    
                    # Find nearest neighbors and estimate smoothness
                    min_dists = np.min(dists, axis=1)
                    
                    # Simple heuristic: if average distance is small, function might be smooth
                    avg_dist = np.mean(min_dists)
                    if avg_dist < 1.0:
                        self.current_kernel_type = "rbf"
                    else:
                        self.current_kernel_type = "matern"
                except:
                    pass
        
        # Select kernel based on current type
        if self.current_kernel_type == "rbf":
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        else:
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=3,  # Slightly more restarts for better optimization
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function with dynamic exploration weight"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using hybrid approach"""
        # Initialize with a good set of starting points
        n_starting_points = min(100 * self.dim, 2000)  # Reduced from previous version for efficiency
        
        # Generate candidates with some diversity consideration
        candidates = []
        
        # Add some random samples for exploration
        if len(candidates) < n_starting_points // 2:
            random_samples = self._sample_points(n_starting_points // 2)
            candidates.extend(random_samples)
            
        # Add points around best known point for exploitation
        if self.X is not None and len(self.X) > 0:
            best_x = self.X[np.argmin(self.y)]
            # Sample around the best point with some randomization
            for _ in range(n_starting_points // 2):
                sample = best_x + (self.rng.random(self.dim) - 0.5) * 0.5
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                candidates.append(sample)
        else:
            # If no points yet, use random sampling
            candidates.extend(self._sample_points(n_starting_points // 2))
            
        candidates = np.array(candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates with diversity consideration
        if batch_size == 1:
            # Simple selection of best point
            selected_idx = np.argmax(acq_values.flatten())
            selected_points = candidates[selected_idx: selected_idx+1]
        else:
            # For batch selection, use a greedy approach to select diverse points
            selected_points = []
            remaining_candidates = candidates.copy()
            remaining_acq = acq_values.copy()
            
            for _ in range(min(batch_size, len(remaining_candidates))):
                if len(remaining_candidates) == 0:
                    break
                    
                # Select the point with highest acquisition value
                best_idx = np.argmax(remaining_acq.flatten())
                selected_points.append(remaining_candidates[best_idx])
                
                # Remove this point and its neighbors from consideration
                if len(remaining_candidates) > 1:
                    distances = cdist([selected_points[-1]], remaining_candidates)[0]
                    # Remove points that are too close
                    mask = distances > 0.1  # Threshold for diversity
                    remaining_candidates = remaining_candidates[mask]
                    remaining_acq = remaining_acq[mask]
                else:
                    break
                    
            selected_points = np.array(selected_points)
            
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

    def _check_model_update_condition(self):
        """Check if we should update the model structure"""
        return (self.iteration_count > 0 and 
                self.iteration_count % self.model_update_frequency == 0)

    def _check_model_reinit_condition(self):
        """Check if we should reinitialize the model"""
        return (self.iteration_count > 0 and 
                self.iteration_count % self.model_reinit_frequency == 0)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol sequence design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            self.iteration_count += 1
            
            # Update the model with new data
            if self.n_evals > self.n_init:
                # Reinitialize model periodically to avoid getting stuck
                if self._check_model_reinit_condition():
                    self.gp_model = self._fit_gp_model(self.X, self.y)
                else:
                    self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Update exploration weight (decrease over time)
            self.exploration_weight *= self.exploration_decay
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
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
  File "<EvolvingSurrogateBO>", line 251, in __call__
 251->             next_points = self._select_next_points(self.batch_size)
  File "<EvolvingSurrogateBO>", line 155, in _select_next_points
 155->         acq_values = self._acquisition_function(candidates)
  File "<EvolvingSurrogateBO>", line 123, in _acquisition_function
 121 |         improvement = y_min - mu
 122 |         z = improvement / sigma
 123->         ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
 124 |         
 125 |         return ei.reshape(-1, 1)
NameError: name 'norm' is not defined
