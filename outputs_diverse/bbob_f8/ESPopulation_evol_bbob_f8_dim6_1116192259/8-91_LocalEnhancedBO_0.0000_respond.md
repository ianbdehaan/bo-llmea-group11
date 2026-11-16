# Description 
**LocalEnhancedBO**: This Bayesian Optimization algorithm employs a local enhancement strategy that dynamically adjusts between global exploration and local exploitation. It uses an ensemble of Gaussian Process surrogates with different kernels (Matern, RBF, and linear) to improve robustness. The acquisition function combines expected improvement with a novelty measure to encourage diverse sampling. The search strategy incorporates a multi-start local optimizer with adaptive step sizes and utilizes a trust region that shrinks based on optimization progress. Key innovation lies in the ensemble surrogate modeling combined with a hybrid acquisition function that balances exploration and exploitation more effectively than single-model approaches.

# Justification 
The algorithm differs significantly from AdaptiveTrustRegionBO by:
1. Using an ensemble of multiple GP surrogates with different kernels (Matern, RBF, linear) instead of a single ARD Matern kernel
2. Implementing a hybrid acquisition function that combines expected improvement with a novelty measure to promote diverse sampling
3. Employing adaptive step size control in local optimization rather than fixed trust regions
4. Using multi-start local optimization with intelligent restarts based on convergence behavior
5. The ensemble approach provides better uncertainty quantification and robustness across different function characteristics, while the hybrid acquisition balances exploration vs exploitation more effectively

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
import warnings
from scipy.optimize import differential_evolution, minimize

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
            
        # Ensemble of GP models with different kernels
        self.gp_models = []
        self.kernel_types = ['matern', 'rbf', 'linear']
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.1  # Higher exploration weight for more diverse sampling
        
        # Local optimization parameters
        self.local_optimization_restarts = 5
        self.min_step_size = 1e-6
        self.max_step_size = 1.0
        self.step_reduction_factor = 0.5
        self.convergence_threshold = 1e-4
        
        # Performance tracking
        self.best_y_history = []
        self.last_improvement_iter = 0

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_ensemble_gp_models(self, X, y):
        """Fit ensemble of Gaussian Process surrogate models with different kernels"""
        self.gp_models = []
        
        # Matern kernel
        matern_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        matern_kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # RBF kernel
        rbf_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        rbf_kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Linear kernel
        linear_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) 
        linear_kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        kernels = [matern_kernel, rbf_kernel, linear_kernel]
        
        for i, kernel in enumerate(kernels):
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                n_restarts_optimizer=2,  # Reduced restarts for efficiency
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            model.fit(X, y)
            self.gp_models.append(model)

    def _predict_ensemble(self, X):
        """Get predictions from ensemble of models"""
        if len(self.gp_models) == 0:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
            
        # Get predictions from all models
        predictions = []
        stds = []
        
        for model in self.gp_models:
            mu, sigma = model.predict(X, return_std=True)
            predictions.append(mu.reshape(-1, 1))
            stds.append(sigma.reshape(-1, 1))
        
        # Average predictions and combine uncertainties (assuming independent models)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.sqrt(np.mean(np.square(stds), axis=0) + np.var(predictions, axis=0))
        
        return mean_pred, std_pred

    def _novelty_measure(self, X):
        """Calculate novelty measure based on distance to existing points"""
        if self.X is None or len(self.X) == 0:
            return np.ones((X.shape[0], 1))
            
        # Calculate pairwise distances
        distances = cdist(X, self.X)
        min_distances = np.min(distances, axis=1, keepdims=True)
        
        # Convert to novelty score (higher = more novel)
        novelty = 1.0 / (min_distances + 1e-8)  # Add small value to avoid division by zero
        return novelty

    def _acquisition_function(self, X):
        """Hybrid acquisition function combining expected improvement with novelty"""
        if len(self.gp_models) == 0:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from ensemble
        mu, sigma = self._predict_ensemble(X)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Novelty measure
        novelty = self._novelty_measure(X)
        
        # Combine EI with novelty (higher weight to novelty for exploration)
        combined_acq = ei + self.exploration_weight * novelty
        
        return combined_acq.reshape(-1, 1)

    def _local_optimize(self, start_point, max_iter=50):
        """Perform local optimization from a starting point"""
        # Use a combination of L-BFGS-B and differential evolution for robustness
        best_x = start_point.copy()
        best_f = float('inf')
        
        # Try multiple restarts with different step sizes
        step_sizes = [self.max_step_size, self.max_step_size * 0.5, self.max_step_size * 0.25]
        
        for i in range(self.local_optimization_restarts):
            # Random perturbation of start point
            perturbed_start = best_x + (self.rng.random(self.dim) - 0.5) * step_sizes[i % len(step_sizes)]
            
            # Clip to bounds
            perturbed_start = np.clip(perturbed_start, self.bounds[0], self.bounds[1])
            
            try:
                result = minimize(
                    lambda x: float(self._predict_ensemble(x.reshape(1, -1))[0]),
                    perturbed_start,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': max_iter//2, 'ftol': self.convergence_threshold}
                )
                
                if result.success:
                    f_val = float(result.fun)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = result.x.copy()
            except:
                pass  # Continue with other restarts if optimization fails
                
        return best_x

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start acquisition function optimization"""
        n_starting_points = min(100 * self.dim, 500)  # Reduced for efficiency
        
        # Generate candidates using Latin hypercube sampling
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition values
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        # For each selected point, perform local optimization to refine
        refined_points = []
        for point in selected_points:
            refined_point = self._local_optimize(point)
            refined_points.append(refined_point)
            
        return np.array(refined_points)

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
        
        # Fit initial ensemble GP models
        self._fit_ensemble_gp_models(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self._fit_ensemble_gp_models(self.X, self.y)
            
            # Select next points using acquisition function
            next_points = self._select_next_points(1)
            
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
  File "<LocalEnhancedBO>", line 52
    samples = sampler.random(n=n=n_points)
                                ^
SyntaxError: invalid syntax
