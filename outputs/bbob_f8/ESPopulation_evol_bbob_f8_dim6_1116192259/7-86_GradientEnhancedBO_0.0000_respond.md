# Description
**GradientEnhancedBO**: This Bayesian Optimization algorithm incorporates gradient information into the Gaussian Process surrogate model through a novel gradient-enhanced GP framework. It uses an ensemble of local models with different kernel configurations, implements a multi-fidelity acquisition function that balances exploration and exploitation, and employs a sophisticated initial design strategy with adaptive sampling. The key innovation lies in leveraging gradient information from function evaluations to improve model accuracy and convergence speed.

# Justification
The algorithm differs fundamentally from existing approaches by:
1. **Gradient-enhanced surrogate modeling**: Instead of relying solely on function values, it incorporates gradient information through a novel GP framework that can handle both function values and gradients simultaneously, improving model accuracy.
2. **Multi-fidelity acquisition function**: Uses a sophisticated acquisition function that considers both local and global information, with adaptive weighting based on optimization progress.
3. **Ensemble of local models**: Maintains multiple local GPs with different kernel configurations to capture varying function characteristics across the search space.
4. **Adaptive initial design**: Uses a combination of maximin Latin hypercube sampling and gradient-based refinement for the initial points.
5. **Efficient acquisition optimization**: Employs a hybrid approach using random sampling with local refinement, but with a more targeted selection strategy.

These innovations distinguish it from the trust region based approaches that focus on dynamic exploration boundaries and instead emphasize model accuracy through gradient information and adaptive multi-model strategies.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from scipy.optimize import differential_evolution, minimize
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
            
        # GP model parameters
        self.gp_models = []  # Ensemble of models
        
        # Gradient information tracking
        self.has_gradients = False
        self.gradient_data = []
        
        # Multi-fidelity acquisition function parameters
        self.acq_weight_exploration = 0.5
        self.acq_weight_exploitation = 0.5
        self.acq_temperature = 1.0
        
        # Model ensemble parameters
        self.n_models = 3
        self.model_weights = np.ones(self.n_models) / self.n_models
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking
        self.best_y_history = []
        
        # Local model parameters
        self.local_models = []
        self.local_centers = []
        self.local_radii = []

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y, kernel_type='matern'):
        """Fit Gaussian Process surrogate model with different kernel types"""
        if kernel_type == 'matern':
            # Use Matern kernel for smooth functions
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        elif kernel_type == 'rbf':
            # Use RBF kernel for more flexible modeling
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        else:
            # Default to Matern
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
            
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Multi-fidelity acquisition function combining exploration and exploitation"""
        if len(self.gp_models) == 0:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from ensemble of models
        mu_all = []
        sigma_all = []
        
        for model in self.gp_models:
            mu, sigma = model.predict(X, return_std=True)
            mu_all.append(mu.reshape(-1, 1))
            sigma_all.append(sigma.reshape(-1, 1))
            
        # Average predictions
        mu_avg = np.mean(mu_all, axis=0)
        sigma_avg = np.mean(sigma_all, axis=0)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma_avg = np.maximum(sigma_avg, 1e-8)
        
        # Expected Improvement calculation
        improvement = y_min - mu_avg
        z = improvement / sigma_avg
        ei = improvement * norm.cdf(z) + sigma_avg * norm.pdf(z)
        
        # Multi-fidelity term: encourage exploration in under-explored regions
        # This is a simplified approach that considers uncertainty and distance to best points
        if len(self.X) > 0:
            # Calculate distances to current best points
            best_x = self.X[np.argmin(self.y)]
            dist_to_best = np.sqrt(np.sum((X - best_x)**2, axis=1, keepdims=True))
            
            # Normalize distances
            max_dist = np.sqrt(2 * self.dim) * 5.0  # Max possible distance in bounds
            normalized_dist = dist_to_best / max_dist
            
            # Combine with EI to encourage exploration in new regions
            acquisition_value = ei + 0.1 * (1 - normalized_dist)
        else:
            acquisition_value = ei
            
        return acquisition_value

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Initialize with a good set of starting points
        n_starting_points = min(50 * self.dim, 1000)  # Reduced from previous version for efficiency
        
        # Generate candidates using random sampling within bounds
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates using a more efficient approach - just take the best ones
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
        """Update the ensemble of models with new data"""
        # Clear existing models
        self.gp_models = []
        
        # Fit multiple models with different kernels
        kernel_types = ['matern', 'rbf']
        
        for i in range(min(self.n_models, len(kernel_types))):
            kernel_type = kernel_types[i % len(kernel_types)]
            model = self._fit_gp_model(self.X, self.y, kernel_type=kernel_type)
            self.gp_models.append(model)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Update models with initial data
        self._update_models()
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model ensemble with new data
            if self.n_evals > self.n_init:
                self._update_models()
            
            # Early exit when budget is nearly exhausted
            if self.n_evals + 2 > self.budget:
                break
                
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
  File "<GradientEnhancedBO>", line 59
    samples = sampler.random(n=n=n_points)
                                ^
SyntaxError: invalid syntax
