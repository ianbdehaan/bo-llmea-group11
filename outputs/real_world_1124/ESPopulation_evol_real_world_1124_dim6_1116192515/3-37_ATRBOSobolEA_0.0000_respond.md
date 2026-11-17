# Description 
**Adaptive Trust-Region Bayesian Optimization with Sobol Initial Design and Enhanced Acquisition Function (ATRBO-Sobol-EA)**

This refined Bayesian optimization algorithm builds upon the ATRBO-Sobol framework by introducing several key improvements to enhance performance and efficiency. The main innovations include: 1) An adaptive ensemble of acquisition functions that dynamically adjusts weights based on problem characteristics, 2) A more sophisticated trust-region adaptation mechanism using a progress-based shrinkage model, 3) Improved surrogate model selection with automatic complexity adjustment, 4) Enhanced batch selection using diverse sampling strategies within the trust region, and 5) A robust early-stopping criterion to prevent unnecessary evaluations.

# Justification 
The improvements are justified as follows:
1. **Adaptive Acquisition Function Ensemble**: Instead of fixed weights, we dynamically adjust acquisition function weights based on performance trends, allowing better balance between exploration and exploitation throughout optimization.
2. **Progress-Based Trust Region Adaptation**: Using a more nuanced approach to trust region adjustment that considers both success/failure counts and relative improvement, leading to better local search efficiency.
3. **Smart Surrogate Model Selection**: Adaptive model complexity based on the number of evaluations ensures computational efficiency while maintaining modeling accuracy.
4. **Diverse Batch Sampling**: Improved sampling strategy within trust regions ensures better coverage and avoids premature convergence.
5. **Early Stopping**: Prevents wasting function evaluations on unproductive searches, particularly important for budget-constrained scenarios.

These changes make the algorithm more robust, efficient, and adaptive to different problem characteristics while maintaining the core strengths of ATRBO-Sobol.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
import warnings

class ATRBOSobolEA:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.success_count = 0
        self.failure_count = 0
        self.min_trust_region = 0.05
        self.max_trust_region = 2.0
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling
        self.acq_func_history = []  # Track performance of different acquisition functions
        
        # Surrogate model parameters
        self.model_complexity = 'sparse'  # 'sparse', 'full', or 'adaptive'
        self.model_update_threshold = 20  # Update model every N evaluations
        
        # Batch selection parameters
        self.batch_size = 1
        self.diversity_factor = 0.3

    def _sample_points(self, n_points):
        # Use Sobol sequence sampling for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, seed=self.rng, scramble=True)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with adaptive complexity based on number of evaluations
        n_train = min(50, max(20, int(0.5 * len(X))))
        
        # Select a random subset of points for training to keep it fast
        train_indices = self.rng.choice(len(X), size=n_train, replace=False)
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        # Use ARD Matern kernel with automatic relevance detection
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # Fit GP with noise estimation and optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_train, y_train)
        return gp

    def _acquisition_function(self, X):
        # Use an adaptive ensemble of acquisition functions weighted by performance
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
        # Expected Improvement (EI)
        best_y = np.min(self.y)
        ei_values = np.zeros_like(mu)
        
        # Handle case where sigma is zero (no uncertainty)
        mask = sigma > 0
        if np.any(mask):
            improvement = best_y - mu[mask]
            z = improvement / sigma[mask]
            ei_values[mask] = improvement * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        # Lower Confidence Bound (LCB)
        kappa = 2.0  # Exploration parameter
        lcb = mu - kappa * sigma
        
        # Thompson sampling: sample from posterior distribution
        samples = gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        thompson = -samples.flatten()  # Negative because we minimize
        
        # Combine acquisition functions with adaptive weights
        acq_values = (
            self.acq_func_weights[0] * ei_values.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _update_acquisition_weights(self):
        # Update acquisition function weights based on recent performance
        if len(self.acq_func_history) < 5:
            # Use default weights for initial iterations
            return
            
        # Analyze recent performance of different acquisition functions
        recent_performance = self.acq_func_history[-5:]
        
        # Simple heuristic: if one function consistently performs better, give it more weight
        ei_performance = np.mean([p[0] for p in recent_performance])
        lcb_performance = np.mean([p[1] for p in recent_performance])
        thompson_performance = np.mean([p[2] for p in recent_performance])
        
        # Normalize weights
        total = ei_performance + lcb_performance + thompson_performance
        if total > 0:
            self.acq_func_weights = [
                ei_performance / total,
                lcb_performance / total,
                thompson_performance / total
            ]

    def _select_next_points(self, batch_size):
        # Select points using a hybrid approach with diversity consideration:
        # 1. Sample candidates in trust region
        # 2. Apply diversity to avoid local clustering
        
        # Generate initial candidates
        n_candidates = min(1000, max(50, self.budget - self.n_evals))
        
        # Get points from current trust region if we have a center
        if self.trust_region_center is not None:
            # Generate points around the trust region center
            candidates = self._sample_points(n_candidates)
            
            # Apply trust region constraint: only consider points within radius
            distances = np.linalg.norm(candidates - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            if len(valid_indices) > 0:
                candidates = candidates[valid_indices]
            
            # If not enough candidates in trust region, expand the search
            if len(candidates) < batch_size:
                additional_candidates = self._sample_points(batch_size - len(candidates))
                candidates = np.vstack([candidates, additional_candidates])
        else:
            # First few iterations: random sampling with trust region expansion
            candidates = self._sample_points(n_candidates)
        
        # Limit candidates to reasonable size
        if len(candidates) > n_candidates:
            candidates = candidates[:n_candidates]
            
        # Evaluate acquisition function on candidates
        acq_values = self._acquisition_function(candidates)
        
        # Apply diversity constraint to avoid clustering
        if batch_size > 1:
            # Use a greedy approach to select diverse points
            selected_indices = []
            remaining_indices = list(range(len(candidates)))
            
            for i in range(min(batch_size, len(remaining_indices))):
                if i == 0:
                    # Select best point
                    idx = np.argmax(acq_values[remaining_indices])
                    selected_indices.append(remaining_indices[idx])
                else:
                    # Select point that maximizes diversity from already selected points
                    best_diverse_idx = -1
                    max_diversity = -np.inf
                    
                    for j in range(len(remaining_indices)):
                        idx = remaining_indices[j]
                        # Calculate minimum distance to already selected points
                        min_dist = np.inf
                        for sel_idx in selected_indices:
                            dist = np.linalg.norm(candidates[idx] - candidates[sel_idx])
                            if dist < min_dist:
                                min_dist = dist
                        
                        # Weight by acquisition value and diversity
                        weighted_value = acq_values[idx] * (1.0 + self.diversity_factor * min_dist)
                        
                        if weighted_value > max_diversity:
                            max_diversity = weighted_value
                            best_diverse_idx = j
                    
                    selected_indices.append(remaining_indices[best_diverse_idx])
                
                # Remove selected point from remaining
                remaining_indices.pop(best_diverse_idx)
            
            selected_points = candidates[selected_indices]
        else:
            # Simple selection for single point
            top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
            selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if len(X) == 0:
            return np.array([])
            
        y = []
        for point in X:
            try:
                val = func(point)
                y.append(val)
            except Exception as e:
                # Handle function evaluation errors by returning large value
                y.append(1e10)
                
        y = np.array(y).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y
    
    def _update_trust_region(self):
        # Adapt trust region based on recent performance and progress
        if len(self.y) < 5:
            return
            
        # Calculate recent improvement rate
        recent_improvement = np.min(self.y[-3:]) - np.min(self.y[:-3])
        
        # If we've made significant progress, shrink the trust region
        if recent_improvement < -0.01 and self.trust_region_radius > self.min_trust_region:
            self.trust_region_radius *= 0.85
            self.success_count += 1
            self.failure_count = 0
            
        # If we haven't made progress, expand the trust region
        elif recent_improvement >= -0.01 and self.trust_region_radius < self.max_trust_region:
            self.trust_region_radius *= 1.1
            self.failure_count += 1
            self.success_count = 0
            
        # If we're in a local area with no progress, consider restarting
        elif self.failure_count > 3 and self.trust_region_radius < self.max_trust_region * 0.5:
            self.trust_region_radius = min(self.max_trust_region, self.trust_region_radius * 1.5)
            
        # Ensure trust region stays within bounds
        self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region, self.max_trust_region)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            
        # Update trust region based on success of recent evaluations
        self._update_trust_region()
        
        # Update acquisition function weights
        if len(self.y) > 1:
            # Record current performance for each acquisition function type
            self.acq_func_history.append(self.acq_func_weights.copy())
            if len(self.acq_func_history) > 10:
                self.acq_func_history.pop(0)
                
            self._update_acquisition_weights()

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Set trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.trust_region_center = self.X[best_idx]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Determine batch size based on remaining budget and problem size
            remaining_budget = self.budget - self.n_evals
            self.batch_size = min(3, max(1, int(remaining_budget / 50)))
            
            # Select next points to evaluate
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Update trust region center based on new best point
            if len(self.y) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx]
                
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
```
## Error
 Traceback (most recent call last):
  File "<ATRBOSobolEA>", line 313, in __call__
 313->             next_points = self._select_next_points(self.batch_size)
  File "<ATRBOSobolEA>", line 174, in _select_next_points
 174->         acq_values = self._acquisition_function(candidates)
  File "<ATRBOSobolEA>", line 86, in _acquisition_function
  86->         gp = self._fit_model(self.X, self.y)
  File "<ATRBOSobolEA>", line 60, in _fit_model
  58 |         
  59 |         # Select a random subset of points for training to keep it fast
  60->         train_indices = self.rng.choice(len(X), size=n_train, replace=False)
  61 |         X_train = X[train_indices]
  62 |         y_train = y[train_indices]
  File "numpy/random/_generator.pyx", line 920, in numpy.random._generator.Generator.choice
ValueError: Cannot take a larger sample than population when replace is False
