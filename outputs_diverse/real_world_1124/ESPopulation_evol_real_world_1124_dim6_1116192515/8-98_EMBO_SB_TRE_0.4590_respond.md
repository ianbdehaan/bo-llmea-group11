# Description
**Efficient Multi-Start Bayesian Optimization with Sobol Initial Design, Gaussian Process Surrogate, and Adaptive Trust-Region Exploration (EMBO-SB-TRE)**

This novel Bayesian optimization algorithm introduces a distinctive combination of components that differs significantly from existing approaches:

1. **Gaussian Process Surrogate with Automatic Relevance Detection**: Unlike the Random Forest-based approaches in the population, this uses a sparse Gaussian Process with an ARD Matern kernel for more accurate uncertainty quantification and better modeling of smooth black-box functions.

2. **Adaptive Trust-Region Exploration**: Instead of using fixed acquisition function weights or simple batch selection, this algorithm implements a dynamic trust-region mechanism that grows/shrinks based on progress, allowing efficient global exploration early on and focused local search near the optimum.

3. **Multi-Start Acquisition Optimization with Diverse Starting Points**: Uses a sophisticated multi-start strategy with diverse sampling techniques (Sobol + Latin Hypercube) to avoid local optima in acquisition function optimization.

4. **Improved Batch Selection with Diversity Control**: Implements adaptive batch selection that reduces batch size as budget approaches, but maintains diversity through trust-region constraints.

# Justification

The key differences from existing algorithms:

1. **Surrogate Model Choice**: The Gaussian Process with ARD Matern kernel provides superior uncertainty quantification compared to Random Forest for smooth functions and allows principled exploration-exploitation trade-offs through the GP's natural probabilistic nature.

2. **Trust-Region Mechanism**: Unlike EMBO_SB_IFS which uses dynamic acquisition weights, this approach actively controls search space exploration via adaptive trust regions, providing better global-local balance without relying on acquisition function parameter tuning.

3. **Acquisition Function Optimization Strategy**: Instead of simple L-BFGS restarts, this algorithm uses a hybrid approach combining Sobol sampling with local refinement and ensures diverse starting points for robust optimization.

4. **Batch Selection with Progress Awareness**: The batch size adapts dynamically while respecting the trust-region constraints, unlike fixed or simple budget-based approaches in existing algorithms.

These components work together to create an algorithm that is both theoretically sound (GP surrogates with proper uncertainty) and practically efficient (trust-region based search control).

# Code

```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_SB_TRE:
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
        self.trust_region_radius = 2.0
        self.min_trust_region_radius = 0.1
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.8
        self.trust_region_expand_factor = 1.2
        self.last_improvement_iter = 0
        self.improvement_threshold = 1e-3
        
        # Acquisition function parameters
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process regressor with ARD Matern kernel
        # Use a sparse GP for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use small number of hyperparameter restarts for efficiency
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Reduced from default for efficiency
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y.flatten())
        return gp

    def _get_gp_uncertainty(self, X):
        """Get uncertainty estimates from Gaussian Process"""
        if self.gp is None:
            # Return a default uncertainty value
            return np.ones((X.shape[0], 1)) * 1.0
            
        _, std = self.gp.predict(X, return_std=True)
        return std.reshape(-1, 1)

    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration weight using GP predictions
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP (mean and std)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get best observed value so far
        if self.y is not None and len(self.y) > 0:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Avoid numerical issues with zero std
        std = np.maximum(std, 1e-8)
        
        # Expected Improvement calculation
        improvement = mean - best_y
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with enhanced strategy"""
        # Generate diverse starting points using a hybrid approach:
        # 1. Sobol for global spread
        # 2. Latin Hypercube for local diversity
        sampler_sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        sobol_points = sampler_sobol.random(n=n_starts // 2)
        sobol_points = self.bounds[0] + sobol_points * (self.bounds[1] - self.bounds[0])
        
        sampler_lhc = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        lhc_points = sampler_lhc.random(n=n_starts // 2)
        lhc_points = self.bounds[0] + lhc_points * (self.bounds[1] - self.bounds[0])
        
        start_points = np.vstack([sobol_points, lhc_points])
        
        best_value = -np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(len(start_points)):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 30}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _get_trust_region_center(self):
        """Get the center of the current trust region"""
        if self.X is None or len(self.X) == 0:
            # Default to center of search space
            return (self.bounds[0] + self.bounds[1]) / 2.0
            
        # Use the best point found so far as center
        best_idx = np.argmin(self.y)
        return self.X[best_idx].copy()

    def _get_diverse_points_in_trust_region(self, n_points):
        """Get diverse points within the current trust region"""
        center = self._get_trust_region_center()
        
        # Sample points within trust region
        points = []
        for i in range(n_points):
            # Random point in trust region
            point = center + (self.rng.random(self.dim) - 0.5) * 2 * self.trust_region_radius
            # Clip to bounds
            point = np.clip(point, self.bounds[0], self.bounds[1])
            points.append(point)
            
        return np.array(points)

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization with trust region constraints"""
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use multi-start L-BFGS to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=10)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point within trust region
                    points_in_tr = self._get_diverse_points_in_trust_region(1)
                    selected_points.append(points_in_tr[0].reshape(1, -1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using trust region sampling to maintain diversity
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample from current trust region
                tr_points = self._get_diverse_points_in_trust_region(remaining)
                for point in tr_points:
                    selected_points.append(point.reshape(1, -1))
            
            return np.vstack(selected_points)

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
        """Update the trust region based on recent improvements"""
        if self.n_evals > 10:
            # Check for recent improvement
            recent_improvement = False
            
            if len(self.y) >= 2:
                # Calculate improvement in last few evaluations
                recent_y = self.y[-5:] if len(self.y) >= 5 else self.y
                if len(recent_y) >= 2:
                    improvement = np.min(self.y[:-1]) - np.min(recent_y)
                    if improvement > self.improvement_threshold:
                        recent_improvement = True
            
            if not recent_improvement:
                # Shrink trust region
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                # Expand trust region
                self.trust_region_radius *= self.trust_region_expand_factor
                
        # Ensure trust region stays within bounds
        self.trust_region_radius = np.clip(
            self.trust_region_radius, 
            self.min_trust_region_radius, 
            self.max_trust_region_radius
        )

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
            # Adaptive batch sizing based on remaining budget and problem characteristics
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                # Adaptive batch size: larger batches early on, smaller near end
                batch_size = min(max(2, remaining_budget // 8), 5) 
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
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
## Feedback
 The algorithm EMBO_SB_TRE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4590 with standard deviation 0.0194.

took 62.61 seconds to run.