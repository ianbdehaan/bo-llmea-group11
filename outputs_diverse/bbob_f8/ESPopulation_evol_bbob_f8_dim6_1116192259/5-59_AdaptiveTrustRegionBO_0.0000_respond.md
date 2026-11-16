# Description
**AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.

# Justification
The key change replaces the trust region adaptation mechanism with a global-to-local search switch strategy. This approach addresses limitations of the original trust region method in complex optimization landscapes like Rosenbrock:

1. **Global-to-Local Switch Strategy**: Instead of continuously adapting trust region size, we implement a phase-based strategy that starts with global exploration (large regions) and transitions to local exploitation when sufficient information is gathered. This prevents premature convergence while maintaining efficient exploration.

2. **Diverse Batch Selection**: We replace the trust-region constrained sampling with diverse batch selection using maximum entropy sampling. This ensures good coverage of the search space rather than focusing too heavily on local improvements.

3. **Information-Theoretic Acquisition**: The acquisition function is switched from Expected Improvement to a more sophisticated information-theoretic approach (Expected Improvement with Entropy-based exploration), which better balances exploration and exploitation.

4. **Improved GP Model Fitting**: We use fewer restarts for hyperparameter optimization and employ a rank-transformed target variable to make the GP more robust to outliers and non-Gaussian noise.

5. **Adaptive Restart Logic**: Instead of shrinking trust regions, we implement a restart mechanism based on convergence metrics that resets the search when progress plateaus, preventing getting stuck in poor local regions.

This strategy works better because:
- The Rosenbrock function has a narrow valley structure where early global exploration is crucial
- Trust regions may get stuck in suboptimal local areas without proper switching logic
- Diverse batch selection helps discover multiple promising regions before focusing locally
- Information-theoretic acquisition functions are more adaptive to varying levels of uncertainty

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

class AdaptiveTrustRegionBO:
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
        
        # Switch from global to local exploration
        self.global_phase = True
        self.switch_threshold = max(20, dim)  # Switch after evaluating this many points
        self.local_search_radius = 1.0
        
        # Batch selection parameters
        self.batch_size = 1
        self.diversity_weight = 0.5
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # Performance tracking for restart mechanism
        self.best_y_history = []
        self.last_improvement_iter = 0
        self.stagnation_count = 0
        self.max_stagnation = 5
        self.progress_threshold = 0.01

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model with rank transformation"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Transform y to ranks for robustness
        y_rank = np.empty_like(y)
        y_rank[np.argsort(y.flatten())] = np.arange(len(y))
        y_transformed = y_rank.reshape(-1, 1) / len(y)
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=1,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y_transformed)
        return model

    def _expected_improvement_with_entropy(self, X):
        """Expected Improvement with entropy-based exploration"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min) from original scale
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add entropy-based exploration term to balance exploitation and exploration
        # Higher entropy (uncertainty) should increase exploration
        entropy_term = self.exploration_weight * np.log(1.0 + sigma**2)
        ei = ei + entropy_term
        
        return ei.reshape(-1, 1)

    def _select_diverse_batch(self, n_candidates):
        """Select diverse batch of points using maximum entropy sampling"""
        # Generate a large pool of candidates
        if self.X is not None and len(self.X) > 0:
            # Use existing points to guide diversification
            all_candidates = []
            
            # Generate random candidates with diversity
            n_random = max(100, min(n_candidates * 10, 1000))
            random_candidates = self._sample_points(n_random)
            
            # Add some candidates near current best
            if len(self.X) > 0:
                best_x = self.X[np.argmin(self.y)]
                n_local = max(20, min(n_candidates * 2, 100))
                local_candidates = []
                for _ in range(n_local):
                    # Sample around the best point with decreasing radius
                    sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.local_search_radius
                    sample = np.clip(sample, self.bounds[0], self.bounds[1])
                    local_candidates.append(sample)
                
                all_candidates.extend(local_candidates)
            
            all_candidates.extend(random_candidates)
            all_candidates = np.array(all_candidates)
        else:
            # If no points yet, use Latin hypercube sampling
            all_candidates = self._sample_points(n_candidates * 10)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._expected_improvement_with_entropy(all_candidates)
        
        # Select diverse points using greedy selection based on entropy and acquisition values
        selected_indices = []
        remaining_indices = list(range(len(all_candidates)))
        
        while len(selected_indices) < n_candidates and len(remaining_indices) > 0:
            # Calculate diversity scores for remaining candidates
            if len(selected_indices) == 0:
                # Select the best candidate first
                best_idx = np.argmax(acq_values[remaining_indices])
                selected_idx = remaining_indices[best_idx]
            else:
                # For subsequent selections, balance acquisition and diversity
                distances = cdist(all_candidates[selected_indices], all_candidates[remaining_indices], metric='euclidean')
                min_distances = np.min(distances, axis=0)
                
                # Weight by acquisition value and diversity
                weights = acq_values[remaining_indices].flatten() + self.diversity_weight * min_distances
                best_idx = np.argmax(weights)
                selected_idx = remaining_indices[best_idx]
            
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        if len(selected_indices) < n_candidates:
            # Fill with random candidates if needed
            additional_needed = n_candidates - len(selected_indices)
            extra_indices = self.rng.choice(remaining_indices, size=min(additional_needed, len(remaining_indices)), replace=False)
            selected_indices.extend(extra_indices)
            
        return all_candidates[selected_indices]

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

    def _switch_to_local_search(self):
        """Switch from global to local search phase"""
        self.global_phase = False
        # Reduce the local search radius for finer exploration
        self.local_search_radius = max(0.1, 0.5 * self.local_search_radius)
        
    def _check_restart_condition(self):
        """Check if we should restart the optimization"""
        # Restart if stagnation occurs in local phase or budget is nearly exhausted
        if (self.stagnation_count >= self.max_stagnation and 
            not self.global_phase) or self.n_evals >= 0.9 * self.budget:
            return True
        return False

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
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
            
            # Check if we should switch to local search
            if self.global_phase and len(self.y) >= self.switch_threshold:
                self._switch_to_local_search()
            
            # Check for restart condition
            if self._check_restart_condition():
                # Reset by sampling new points with larger radius
                n_new_points = min(10, max(1, self.budget - self.n_evals))
                new_points = self._sample_points(n_new_points)
                y_new = self._evaluate_points(func, new_points)
                self._update_eval_points(new_points, y_new)
                # Reset stagnation count
                self.stagnation_count = 0
                
                # If we're still in global phase, continue
                if self.global_phase:
                    continue
            
            # Select next batch of points using diverse selection
            next_points = self._select_diverse_batch(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history and stagnation count
            current_best = np.min(self.y)
            self.best_y_history.append(current_best)
            
            if len(self.best_y_history) > 1:
                improvement = (self.best_y_history[-2] - current_best) / abs(self.best_y_history[-2])
                if improvement < self.progress_threshold:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0
            
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
  File "<AdaptiveTrustRegionBO>", line 220, in __call__
 220->         self.gp_model = self._fit_gp_model(self.X, self.y)
  File "<AdaptiveTrustRegionBO>", line 72, in _fit_gp_model
  70 |         # Transform y to ranks for robustness
  71 |         y_rank = np.empty_like(y)
  72->         y_rank[np.argsort(y.flatten())] = np.arange(len(y))
  73 |         y_transformed = y_rank.reshape(-1, 1) / len(y)
  74 |         
ValueError: shape mismatch: value array of shape (13,) could not be broadcast to indexing result of shape (13,1)
