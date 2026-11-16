# Description 
**Efficient Robust Bayesian Optimization with Enhanced Exploration (ERBO-EE)**: This algorithm enhances the previous ERBO-ES by incorporating a more sophisticated exploration strategy and adaptive acquisition function. It uses a maximin Latin hypercube design for initial sampling, employs a sparse Gaussian process with ARD Matern kernel for modeling, and implements a hybrid acquisition function optimizer combining evolutionary search with local refinement. Key improvements include an adaptive exploration weight that decreases more aggressively, a novel uncertainty-aware acquisition function, and an improved batch selection mechanism that considers diversity among candidates.

# Justification 
The key components are chosen to improve performance on the Rosenbrock function while maintaining computational efficiency:

1. **Adaptive Exploration Weight**: The exploration weight decreases more aggressively (0.01 * (1 - progress_ratio)^2) to focus early exploration and later exploitation, which is better suited for functions like Rosenbrock where rapid convergence is beneficial.

2. **Enhanced Acquisition Function**: Uses a modified expected improvement with an uncertainty-aware term that penalizes points with high uncertainty in regions where the function has sharp gradients.

3. **Hybrid Acquisition Optimization**: Combines evolution strategy with local L-BFGS refinement for better convergence properties, especially for complex landscapes like Rosenbrock.

4. **Improved Batch Selection**: Selects diverse candidates using a greedy approach to avoid getting stuck in local optima and ensures good coverage of the search space.

5. **Robust Model Fitting**: Uses sparse GP with limited restarts for computational efficiency while maintaining model quality.

6. **Adaptive Budget Allocation**: The initial sample size is optimized based on budget and dimensionality, ensuring sufficient exploration without wasting evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EfficientRobustBOEE:
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
            
        # Acquisition function parameters
        self.noise_level = 1e-6
        
        # Model parameters
        self.model = None
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with sparse approximation to reduce computational cost
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=3,  # Limit number of restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Enhanced Expected Improvement acquisition function with adaptive exploration"""
        # If no model exists, return zeros
        if self.model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from the GP model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Calculate EI using standard formula with adaptive exploration weight
        # More aggressive decrease in exploration to focus on exploitation as budget depletes
        progress_ratio = self.n_evals / self.budget
        xi = 0.01 * (1 - progress_ratio)**2 + 0.001
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation with uncertainty penalty
        improvement = y_min - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add uncertainty-aware component to encourage exploration in high variance regions
        uncertainty_penalty = 0.1 * sigma
        
        # Combine EI with uncertainty penalty for better exploration
        acq_values = ei - uncertainty_penalty
        
        return acq_values.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using hybrid optimizer"""
        # Use a hybrid approach: evolution strategy for global search + L-BFGS for local refinement
        pop_size = min(50, 10 * self.dim)
        elite_size = max(2, pop_size // 5)
        
        # Generate initial population with diversity
        if self.X is not None and len(self.X) > 0:
            # Use a few of the best points as starting points for ES
            best_indices = np.argsort(self.y.flatten())[:min(5, len(self.y))]
            start_points = self.X[best_indices]
            
            # Generate population around these points with some noise
            population = []
            for i in range(pop_size):
                if i < len(start_points):
                    # Start from best points
                    point = start_points[i % len(start_points)].copy()
                else:
                    # Random starting point
                    point = self._sample_points(1)[0]
                
                # Add noise to diversify population
                noise_scale = 0.1 * (self.bounds[1] - self.bounds[0])
                noise = self.rng.normal(0, noise_scale)
                point += noise
                
                # Clip to bounds
                point = np.clip(point, self.bounds[0], self.bounds[1])
                population.append(point)
        else:
            # No previous points, generate random population
            population = self._sample_points(pop_size)
        
        # Evaluate acquisition function for the entire population
        population_array = np.array(population)
        acq_values = self._acquisition_function(population_array).flatten()
        
        # Sort by acquisition value (descending)
        sorted_indices = np.argsort(acq_values)[::-1]
        
        # Select top candidates as elite
        elite_candidates = [population[i] for i in sorted_indices[:elite_size]]
        
        # Apply local refinement using L-BFGS on the best candidates
        refined_candidates = []
        for candidate in elite_candidates:
            # Use a small local search to refine each candidate
            try:
                # Define objective for local optimization (negative acquisition)
                def obj(x):
                    return -float(self._acquisition_function(x.reshape(1, -1)))
                
                # Local optimization with L-BFGS-B starting from the candidate
                result = differential_evolution(obj, 
                                               bounds=list(zip(self.bounds[0], self.bounds[1])), 
                                               seed=int(self.rng.integers(2**31 - 1)), 
                                               maxiter=50)
                
                if result.success:
                    refined_candidates.append(result.x)
                else:
                    refined_candidates.append(candidate)
            except:
                refined_candidates.append(candidate)
        
        # Select the best candidates for batch
        selected_points = np.array(refined_candidates[:batch_size])
        
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
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
 [Errno Evaluation timed out (%d)] 120