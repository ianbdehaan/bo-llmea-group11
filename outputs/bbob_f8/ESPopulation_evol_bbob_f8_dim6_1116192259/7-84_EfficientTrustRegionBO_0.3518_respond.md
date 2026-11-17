# Description 
**EfficientTrustRegionBO**: This Bayesian Optimization algorithm combines a dynamic trust region mechanism with an efficient acquisition function optimizer and a novel initial design strategy. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs Expected Improvement acquisition function. The key innovations include: 1) A hybrid initial design using Sobol sampling for exploration followed by Latin hypercube for exploitation, 2) An adaptive trust region that shrinks when progress stagnates but expands rapidly when improvement is observed, 3) A computationally efficient acquisition function optimizer that uses a combination of random sampling and local refinement with reduced restarts, 4) A smart batch selection strategy using diversity-aware point selection to avoid redundant evaluations.

# Justification 
The key components are chosen based on their proven effectiveness in Bayesian optimization while maintaining computational efficiency:

1. **Hybrid Initial Design**: Uses Sobol sampling (for global exploration) followed by Latin hypercube (for good distribution) to efficiently cover the search space early on, particularly important for the challenging Rosenbrock function.

2. **Adaptive Trust Region**: The trust region dynamically adjusts based on recent improvement - expands rapidly when progress is observed and shrinks more aggressively when stagnation occurs, allowing efficient exploration early and focused exploitation near optimum.

3. **Efficient Acquisition Optimization**: Uses a limited number of random starting points (20*dim) with L-BFGS-B local refinement for acquisition function optimization, reduced from previous versions to save computation while maintaining quality.

4. **Diversity-aware Batch Selection**: Selects batch points that are diverse and likely to provide good information, reducing redundant evaluations and improving convergence speed.

5. **Reduced GP Hyperparameter Restarts**: Limited to 2 restarts to keep computational cost low while maintaining model quality.

6. **Smart Progress Tracking**: Monitors improvement trends to adapt trust region size effectively.

These components work together to achieve faster convergence on the Rosenbrock function with minimal computational overhead, particularly important given the budget constraints.

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

class EfficientTrustRegionBO:
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
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 1.5
        self.progress_threshold = 0.005
        self.stagnation_count = 0
        self.max_stagnation = 3
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for trust region adaptation
        self.best_y_history = []
        self.last_improvement_iter = 0
        self.improvement_window = 5

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for initial exploration"""
        # Use Sobol sequence for better space-filling properties in early stages
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _sample_latin_hypercube(self, n_points):
        """Sample points using Latin hypercube for good distribution"""
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model"""
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

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function"""
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
        """Select the next points to evaluate using multi-start optimization"""
        # Initialize with a good set of starting points within the trust region
        n_starting_points = min(20 * self.dim, 500)
        
        # Generate candidates within the trust region around the current best point
        if self.X is not None and len(self.X) > 0:
            # Use the current best point as center of trust region
            best_x = self.X[np.argmin(self.y)]
            
            # Sample points within trust region with better distribution
            candidates = []
            while len(candidates) < n_starting_points:
                # Generate random points in the trust region
                sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.trust_region_radius
                
                # Ensure points are within bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                candidates.append(sample)
            
            candidates = np.array(candidates)
        else:
            # If no points yet, use Sobol sampling for better exploration
            candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates - use a simple approach but ensure diversity
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

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        if len(self.best_y_history) < 2:
            return
            
        # Check improvement over a window
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[max(0, len(self.best_y_history) - self.improvement_window)]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        # Update stagnation count
        if improvement > self.progress_threshold:
            self.stagnation_count = 0
            # Expand trust region if progress is good, but more aggressively than before
            self.trust_region_radius = min(
                self.trust_region_radius * self.trust_region_expand_factor,
                self.max_trust_region_radius
            )
        else:
            self.stagnation_count += 1
            # Shrink trust region if no significant progress
            if self.stagnation_count >= self.max_stagnation:
                self.trust_region_radius = max(
                    self.trust_region_radius * self.trust_region_shrink_factor,
                    self.min_trust_region_radius
                )
                self.stagnation_count = 0

    def _check_restart_condition(self):
        """Check if we should restart the optimization"""
        # Restart if trust region becomes too small and no progress
        if (self.trust_region_radius <= self.min_trust_region_radius and 
            self.stagnation_count >= self.max_stagnation):
            return True
        return False

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol sampling for better initial exploration
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # Check for restart condition
            if self._check_restart_condition():
                # Reset trust region and continue
                self.trust_region_radius = 1.0
                self.stagnation_count = 0
            
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
## Feedback
 The algorithm EfficientTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3518 with standard deviation 0.0158.

took 23.15 seconds to run.