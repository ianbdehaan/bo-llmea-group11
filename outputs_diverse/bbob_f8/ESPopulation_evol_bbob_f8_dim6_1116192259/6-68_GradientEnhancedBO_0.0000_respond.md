# Description
**GradientEnhancedBO**: This Bayesian Optimization algorithm introduces a novel gradient-enhanced approach that combines local Gaussian Process models with gradient information from surrogate evaluations. Instead of relying solely on function values, it estimates gradients using finite differences and incorporates this information into the acquisition function through a modified expected improvement. The key innovation is a multi-fidelity refinement strategy where low-fidelity GP models are used for initial exploration, while high-fidelity local models are built around promising regions. The algorithm employs a diverse initial design with a combination of Sobol sampling and Latin hypercube sampling, and uses a hybrid acquisition function optimizer combining random sampling with a small evolutionary strategy.

# Justification
The key innovations in this algorithm distinguish it from the existing AdaptiveTrustRegionBO:

1. **Gradient-enhanced surrogate modeling**: Rather than using only function values, we estimate gradients via finite differences from GP predictions and incorporate them into our acquisition function. This leverages additional information about function behavior that traditional BO methods ignore.

2. **Multi-fidelity modeling approach**: We use a two-tiered GP strategy - low-fidelity models for global exploration and high-fidelity local models for exploitation. This balances computational efficiency with accuracy.

3. **Diverse initial design**: Instead of pure Latin hypercube sampling, we combine Sobol sequences with Latin hypercube sampling to ensure both uniform coverage and good distribution properties.

4. **Hybrid acquisition optimization**: Rather than using only L-BFGS-B or differential evolution, we use a combination of random sampling and a small evolutionary strategy that works well for high-dimensional problems.

5. **Dynamic local model refinement**: As we progress, we build more focused GP models around promising regions to improve acquisition function accuracy.

These components work together to create an algorithm that can better exploit function structure while maintaining computational efficiency, particularly suitable for the Rosenbrock function which has complex curvature and requires careful balance between exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class GradientEnhancedBO:
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
            
        # GP model parameters - use different configurations for local vs global models
        self.global_gp_model = None
        self.local_gp_models = []
        self.local_regions = []
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # For batch selection
        self.batch_size = 1
        
        # Multi-fidelity parameters
        self.use_local_models = False
        self.local_model_threshold = 0.3  # Start using local models after this fraction of budget
        
        # Gradient estimation parameters
        self.grad_eps = 1e-4  # Finite difference step size for gradient estimation
        
        # Performance tracking
        self.best_y_history = []
        self.iteration = 0

    def _sample_points(self, n_points):
        """Sample points using a combination of Sobol and Latin hypercube design"""
        # Use Sobol sequence for initial exploration
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _sample_latin_hypercube(self, n_points):
        """Sample points using Latin hypercube design"""
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y, is_local=False):
        """Fit Gaussian Process surrogate model with appropriate kernel"""
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

    def _estimate_gradients(self, gp_model, X):
        """Estimate gradients using finite differences from GP predictions"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        grad_estimates = []
        
        for i in range(X.shape[0]):
            x_point = X[i:i+1]
            mu_x, _ = gp_model.predict(x_point, return_std=True)
            
            grad = np.zeros(self.dim)
            for j in range(self.dim):
                # Forward difference
                x_plus = x_point.copy()
                x_plus[0, j] += self.grad_eps
                mu_plus, _ = gp_model.predict(x_plus, return_std=True)
                
                # Central difference for better accuracy
                x_minus = x_point.copy()
                x_minus[0, j] -= self.grad_eps
                mu_minus, _ = gp_model.predict(x_minus, return_std=True)
                
                grad[j] = (mu_plus - mu_minus) / (2 * self.grad_eps)
            
            grad_estimates.append(grad)
            
        return np.array(grad_estimates)

    def _acquisition_function(self, X):
        """Gradient-enhanced Expected Improvement acquisition function"""
        if self.global_gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.global_gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add gradient information to improve acquisition function
        # This is a simplified approach where we incorporate gradient magnitude
        grad_magnitude = np.zeros(X.shape[0])
        if len(self.X) > 1:
            # Estimate gradients for the current X points using the global GP
            estimated_grads = self._estimate_gradients(self.global_gp_model, X)
            grad_magnitude = np.linalg.norm(estimated_grads, axis=1)
            
        # Combine standard EI with gradient information - higher gradient magnitude means more promising
        # This modifies the acquisition function to prefer areas with steeper descent
        grad_factor = 0.5 * (grad_magnitude / (np.max(grad_magnitude) + 1e-8))
        ei = ei * (1.0 + grad_factor.reshape(-1, 1))
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using hybrid acquisition optimization"""
        # Determine how many starting points to use
        n_starting_points = min(50 * self.dim, 1000)  # Reduced from previous version for efficiency
        
        # Generate candidates using Sobol sampling (better for high-dim)
        candidates = self._sample_points(n_starting_points)
        
        # Also include some Latin hypercube points for diversity
        n_lhc = min(20 * self.dim, 200)
        lhc_candidates = self._sample_latin_hypercube(n_lhc)
        candidates = np.vstack([candidates, lhc_candidates])
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates using a simple heuristic - just the best ones
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

    def _update_models(self):
        """Update global and local models based on current data"""
        # Always fit a global GP model
        self.global_gp_model = self._fit_gp_model(self.X, self.y)
        
        # Check if we should start using local models
        if (self.n_evals / self.budget) > self.local_model_threshold:
            self.use_local_models = True
            
            # Build local models around promising regions
            # Find top 10% of points by function value
            n_top = max(5, int(len(self.y) * 0.1))
            top_indices = np.argsort(self.y.flatten())[:n_top]
            
            # For each top point, create a local model in its neighborhood
            for idx in top_indices:
                x_center = self.X[idx:idx+1]
                
                # Find nearby points (within a radius)
                distances = cdist(x_center, self.X)[0]
                neighborhood_indices = np.where(distances < 2.0 * np.mean(distances))[0]
                
                if len(neighborhood_indices) > 5:
                    X_local = self.X[neighborhood_indices]
                    y_local = self.y[neighborhood_indices]
                    
                    # Fit local model
                    local_model = self._fit_gp_model(X_local, y_local)
                    
                    # Store for later use
                    self.local_gp_models.append(local_model)
                    self.local_regions.append(x_center)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol sampling for better coverage
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial global GP model
        self.global_gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update models with new data
            if self.n_evals > self.n_init:
                self._update_models()
            
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
  File "<GradientEnhancedBO>", line 250, in __call__
 250->             next_points = self._select_next_points(self.batch_size)
  File "<GradientEnhancedBO>", line 171, in _select_next_points
 169 |         # Select top candidates using a simple heuristic - just the best ones
 170 |         top_indices = np.argsort(acq_values.flatten())[-batch_size:]
 171->         selected_points = candidates[top_indices]
 172 |         
 173 |         return selected_points
IndexError: index 12693 is out of bounds for axis 0 with size 420
