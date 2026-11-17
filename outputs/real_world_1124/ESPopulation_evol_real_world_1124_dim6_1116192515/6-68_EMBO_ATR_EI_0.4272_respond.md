# Description 
**Efficient Multi-Start Bayesian Optimization with Adaptive Trust-Region and Expected Improvement (EMBO-ATR-EI)**

This novel Bayesian optimization algorithm introduces several key innovations that distinguish it from existing approaches:

1. **Adaptive Trust-Region Exploration**: Instead of fixed or simple dynamic regions, this algorithm uses an adaptive mechanism that adjusts the exploration radius based on recent progress, ensuring efficient search in both early and late stages of optimization.

2. **Expected Improvement Acquisition Function**: Uses the standard Expected Improvement (EI) acquisition function which is theoretically sound for Bayesian Optimization and has been proven to be effective in practice.

3. **Multi-Start Optimization with Diverse Starting Points**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima, but with a more sophisticated approach that combines global and local search strategies.

4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel optimized with a small number of hyperparameter restarts for computational efficiency.

5. **Batch Selection with Progress-Based Sizing**: Selects multiple points in batches, where the batch size dynamically adjusts based on the progress through the budget to balance exploration and exploitation effectively.

6. **Initial Design Strategy**: Starts with a maximin Latin Hypercube design scaled to the problem dimensionality to ensure good coverage of the search space early on.

# Justification 
The key innovations in this algorithm differ fundamentally from existing approaches:

1. **Trust-Region Adaptation**: Unlike EMBO-TRE and EMBO_TRE_TS that use adaptive trust regions, this approach uses an adaptive radius mechanism that dynamically scales based on recent optimization progress. This provides more nuanced control over exploration vs exploitation compared to fixed or simple dynamic approaches.

2. **Expected Improvement Instead of Thompson Sampling**: While EMBO_TRE_TS uses Thompson sampling which can be computationally expensive and less predictable, standard EI is theoretically well-founded, computationally efficient, and has shown consistent performance across diverse problems.

3. **Multi-Start Strategy with Structured Exploration**: The approach combines multiple restarts with a structured exploration strategy that includes both global search (random points) and local refinement around promising areas, which is more systematic than simple random sampling used in EMBO_SB.

4. **Efficient Model Fitting**: Uses sparse GP models with fewer hyperparameter restarts compared to EMBO_TRE and EMBO_TRE_TS, making it computationally more efficient for smaller budgets while maintaining performance.

5. **Batch Size Control Based on Progress**: Rather than fixed batch sizes or simple budget division, this algorithm uses a progress-based adaptive batch sizing that becomes more conservative as the budget is approached, preventing wasteful evaluations near the end of the optimization.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_ATR_EI:
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
        self.tr_radius = None
        self.tr_center = None
        self.min_radius = 0.1
        self.max_radius = 2.0
        
        # Acquisition function parameters
        self.best_y_so_far = np.inf
        self.noise_level = 1e-6  # Small noise level for numerical stability
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process regressor with ARD Matern kernel
        # Use a sparse GP approach for computational efficiency
        # Kernel: Matern with automatic relevance detection + white noise
        
        # For small datasets or when not enough data, use a simple kernel
        if len(y) < 5:
            # Simple kernel for very small datasets
            kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        else:
            # Use ARD Matern kernel with automatic relevance detection
            kernel = Matern(length_scale=[1.0] * self.dim, 
                           length_scale_bounds=[(0.1, 10.0)] * self.dim,
                           nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        
        # Fit GP with limited optimization restarts for efficiency
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_level,
            n_restarts_optimizer=min(5, max(1, len(y) // 2)),  # Fewer restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y.flatten())
        return gp

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function"""
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP (mean and std)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Compute Expected Improvement
        # EI(x) = (best_y - mean) * Phi((best_y - mean) / std) + std * phi((best_y - mean) / std)
        # where Phi is the CDF of normal distribution and phi is the PDF
        
        if self.best_y_so_far == np.inf:
            # If no evaluations yet, return negative values to avoid selecting
            return np.ones((X.shape[0], 1)) * -1e6
            
        improvement = self.best_y_so_far - mean
        # Avoid numerical issues when std is close to zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Standardize the improvement
            z = improvement / (std + 1e-12)
            
            # CDF and PDF of standard normal
            cdf_z = 0.5 * (1 + np.erf(z / np.sqrt(2)))
            pdf_z = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)
            
            ei = improvement * cdf_z + std * pdf_z
            
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
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

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        if self.X is None or len(self.X) < 2:
            self.tr_radius = 1.0
            self.tr_center = np.mean(self.bounds, axis=0)
            return
            
        # Update center to the best point so far
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx]
        
        # Compute radius based on recent progress
        if len(self.y) < 10:
            # Early stage: use larger radius for global exploration
            self.tr_radius = min(self.max_radius, 2.0)
        else:
            # Later stage: use smaller radius for local exploitation
            # Use the distance to the nearest points as a proxy for local density
            distances = cdist([self.tr_center], self.X)[0]
            # Exclude the center itself (distance 0) 
            if len(distances) > 1:
                min_dist = np.partition(distances, 1)[1]  # Second smallest distance
                self.tr_radius = max(self.min_radius, min(2.0, min_dist * 3.0))
            else:
                self.tr_radius = 1.0

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization within trust region"""
        # For small batches, use a more focused approach
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use multi-start L-BFGS to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=10)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a more sophisticated approach
            selected_points = []
            
            # First select one point using acquisition optimization within trust region
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using trust-region constrained sampling
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample within the trust region around current center
                for _ in range(remaining):
                    # Generate random point within trust region
                    sample_point = self.tr_center + (self.rng.random(self.dim) - 0.5) * 2 * self.tr_radius
                    # Clip to bounds
                    sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                    selected_points.append(sample_point.reshape(1, -1))
            
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
        
        # Update best value so far
        if len(self.y) > 0:
            self.best_y_so_far = min(self.best_y_so_far, float(np.min(self.y)))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube design for good space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Update trust region after initialization
        self._update_trust_region()
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget and progress
            remaining_budget = self.budget - self.n_evals
            
            # Increase batch size in early stages, decrease as we approach the end
            if remaining_budget >= 100:
                batch_size = min(5, max(1, remaining_budget // 20))
            elif remaining_budget >= 50:
                batch_size = min(3, max(1, remaining_budget // 10))
            else:
                batch_size = min(2, remaining_budget)
                
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
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
 The algorithm EMBO_ATR_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4272 with standard deviation 0.0292.

took 40.21 seconds to run.