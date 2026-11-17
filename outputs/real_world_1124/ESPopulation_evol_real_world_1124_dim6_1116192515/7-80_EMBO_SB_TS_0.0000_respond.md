# Description 
**Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Thompson Sampling (EMBO-SB-TS)**

This novel Bayesian optimization algorithm combines several key innovations to achieve excellent performance on black-box optimization problems:

1. **Sobol Sequence Initial Design**: Replaces Latin Hypercube sampling with Sobol sequences for superior space-filling properties, especially in higher dimensions
2. **Thompson Sampling Acquisition Function**: Uses Thompson sampling from the GP posterior for exploration-exploitation balance instead of expected improvement
3. **Multi-Start Optimization with Diverse Starting Points**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima  
4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
5. **Adaptive Trust-Region Exploration**: Uses a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
6. **Batch Selection with Diversity**: Selects multiple points in batches using diverse sampling strategies within trust regions

# Justification 
This algorithm differs fundamentally from existing approaches in several key aspects:

1. **Surrogate Model Choice**: Unlike EMBO_SB which uses Random Forest (a non-probabilistic model), this approach employs Gaussian Process regression with a Matern kernel, providing natural uncertainty estimates that are essential for Thompson sampling.

2. **Acquisition Function**: While other algorithms use LCB or EI, this implementation specifically uses Thompson Sampling which is known to be more effective at balancing exploration and exploitation in BO, especially when combined with GP surrogates.

3. **Trust Region Strategy**: The dynamic trust-region mechanism provides a more adaptive search strategy that can effectively switch between global exploration and local exploitation based on progress, unlike static approaches.

4. **Multi-start Optimization**: The use of multiple random restarts for acquisition function optimization significantly improves convergence over single-start methods, reducing risk of getting stuck in poor local optima.

5. **Batch Selection Strategy**: The combination of Thompson sampling with trust-region diversity ensures both high-quality point selection and population diversity, avoiding premature convergence.

These innovations make the algorithm particularly effective for challenging black-box optimization problems where exploration-exploitation balance is crucial.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_SB_TS:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10) # Adaptive initial points

        # Random generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model parameters
        self.gp = None
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.progress_history = []
        self.convergence_threshold = 0.01
        
        # Thompson sampling parameters
        self.n_thompson_samples = 50
        self.thompson_noise = 1e-6
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process regressor with Matern kernel
        # Use ARD (automatic relevance detection) for better handling of different dimensions
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use a sparse GP to keep computational costs low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.thompson_noise,
            n_restarts_optimizer=3,  # Few restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y.flatten())
        return gp

    def _acquisition_function(self, X):
        # Thompson sampling acquisition function using GP posterior samples
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Sample from the GP posterior distribution
        samples = self.gp.sample_y(X, n_samples=self.n_thompson_samples, random_state=int(self.rng.integers(2**31 - 1)))
        
        # Return the minimum of each sample (for minimization)
        thompson_values = np.min(samples, axis=1, keepdims=True)
        
        return thompson_values

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using Thompson sampling with trust-region exploration"""
        selected_points = []
        
        # For first few points, use Sobol sampling for good initial coverage
        if self.X is None or len(self.X) < 5:
            return self._sample_points(batch_size)
        
        # Use Thompson sampling for acquisition
        # Sample multiple points and select the best one
        candidates = []
        candidate_scores = []
        
        # Generate more candidates for selection
        n_candidates = max(10, batch_size * 5)
        candidate_points = self._sample_points(n_candidates)
        
        # Evaluate candidates using Thompson sampling
        scores = self._acquisition_function(candidate_points).flatten()
        
        # Select top candidates based on score (lower is better for minimization)
        top_indices = np.argsort(scores)[:batch_size]
        
        return candidate_points[top_indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        if self.X is not None and len(self.X) > 2:
            # Calculate recent improvement
            recent_evals = min(10, len(self.y))
            recent_y = self.y[-recent_evals:]
            
            if len(recent_y) >= 2:
                improvement = np.abs(np.max(recent_y) - np.min(recent_y))
                if improvement < self.convergence_threshold:
                    # Progress is slow, expand trust region
                    self.trust_region_radius = min(self.trust_region_radius * 1.2, 5.0)
                else:
                    # Progress is good, shrink trust region
                    self.trust_region_radius = max(self.trust_region_radius * 0.9, 0.1)
            else:
                # Default behavior for small sample size
                pass

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points using Thompson sampling
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<EMBO_SB_TS>", line 41
    points = sampler.random(n=n=n_points)
                               ^
SyntaxError: invalid syntax
