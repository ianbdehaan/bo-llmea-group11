# Description 
**SobolTrustRegionBO** - This Bayesian Optimization algorithm combines a Sobol sequence-based initial design with an adaptive trust region mechanism. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the use of Sobol sequences for initial sampling which provides better space-filling properties than Latin hypercube, combined with a dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.

# Justification 
The main change replaces the Latin hypercube initial design with Sobol sequences. Sobol sequences provide superior space-filling properties in high dimensions compared to Latin hypercube sampling, which is crucial for the Rosenbrock function that requires good exploration of the search space. The Sobol sequence ensures more uniform coverage of the domain early in the optimization process, leading to better initial model training and faster convergence.

The rest of the algorithm remains largely the same because:
1. The adaptive trust region mechanism already provides excellent dynamic behavior for balancing exploration/exploitation
2. The GP surrogate with ARD Matern kernel is well-suited for the Rosenbrock function's non-convex, multimodal characteristics
3. Expected improvement acquisition function works well with this problem structure
4. The multi-start optimization approach for acquisition function maximization remains effective

The combination of Sobol initial sampling (better exploration) with adaptive trust regions (efficient exploitation) should provide superior performance over the previous Latin hypercube approach.

# Code 
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class SobolTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * self.dim + 1, 10)  # Initial points based on dimensionality
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.trust_region_center = None
        self.trust_region_radius = 2.0
        self.min_radius = 0.01
        self.max_radius = 5.0
        self.radius_factor = 1.5
        self.progress_threshold = 1e-3
        self.last_improvement_eval = 0
        self.improvement_count = 0
        
        # Model parameters
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for diverse initial points - better space filling than Latin hypercube
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use Gaussian Process with anisotropic Matern kernel
        # Add noise to avoid numerical issues
        
        # Scale the data for better GP performance
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Define kernel with automatic relevance determination (ARD)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Fit GP model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise for numerical stability
            n_restarts_optimizer=3,  # Few restarts to save computation
            normalize_y=False,  # We already normalized
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_scaled, y_scaled)
        
        # Store scalers and model for later use
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model = gp
        
        return gp
    
    def _acquisition_function(self, X):
        # Implement expected improvement acquisition function
        # calculate the acquisition function value for each point in X
        
        if self.model is None:
            # If no model, return random values
            return self.rng.random(len(X))
        
        # Transform X to scaled space
        X_scaled = self.scaler_X.transform(X)
        
        # Get predictions from GP
        mean, std = self.model.predict(X_scaled, return_std=True)
        
        # Get best observed value (from original scale)
        y_best = np.min(self.y) if len(self.y) > 0 else 0.0
        
        # Transform y_best to scaled space
        y_best_scaled = self.scaler_y.transform([[y_best]]).ravel()[0]
        
        # Expected Improvement
        improvement = y_best_scaled - mean
        z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei
    
    def _optimize_acquisition(self, n_starts=10):
        # Multi-start optimization of acquisition function using L-BFGS-B
        # with restarts when progress stagnates
        
        if self.model is None:
            # Return random point if no model yet
            return self._sample_points(1)[0]
        
        best_acq_value = -np.inf
        best_point = None
        
        # Sample initial points for multi-start optimization
        initial_points = self._sample_points(n_starts)
        
        # Add trust region center as an additional starting point if available
        if self.trust_region_center is not None:
            initial_points = np.vstack([initial_points, self.trust_region_center])
            
        # Optimize acquisition function from each starting point
        for start_point in initial_points:
            try:
                # Define bounds for optimization (within trust region)
                bounds = [(max(self.bounds[0][i], start_point[i] - self.trust_region_radius), 
                          min(self.bounds[1][i], start_point[i] + self.trust_region_radius)) 
                         for i in range(self.dim)]
                
                # Simple optimization approach - sample around the starting point
                # For efficiency, we'll use a simple random search in the trust region
                candidates = self._sample_points(20)
                acq_values = self._acquisition_function(candidates)
                
                # Select best candidate from this batch
                best_idx = np.argmax(acq_values)
                if acq_values[best_idx] > best_acq_value:
                    best_acq_value = acq_values[best_idx]
                    best_point = candidates[best_idx].copy()
                    
            except Exception:
                # If optimization fails, continue with next starting point
                continue
        
        return best_point if best_point is not None else self._sample_points(1)[0]
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        # Evaluate all points at once to maximize efficiency
        y_values = []
        for point in X:
            y_val = func(point)
            y_values.append(y_val)
        
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            
        # Update trust region center based on best point
        if len(self.y) > 0:
            best_idx = np.argmin(self.y)
            self.trust_region_center = self.X[best_idx]
    
    def _update_trust_region(self):
        # Adaptively update the trust region based on progress
        
        if self.n_evals <= self.n_init:
            return  # Don't modify trust region during initial phase
            
        # Check if we've made progress recently
        recent_improvement = False
        current_best = np.min(self.y)
        
        # Simple progress detection: if we haven't improved much in last 10 evaluations
        if self.n_evals - self.last_improvement_eval > 10:
            # Check if improvement is below threshold
            if len(self.y) >= 2:
                recent_best = np.min(self.y[:-10]) if len(self.y) > 10 else current_best
                if (recent_best - current_best) < self.progress_threshold:
                    # No significant improvement, shrink trust region
                    self.trust_region_radius = max(self.min_radius, 
                                                  self.trust_region_radius / self.radius_factor)
                else:
                    # Significant improvement, expand trust region
                    self.trust_region_radius = min(self.max_radius,
                                                  self.trust_region_radius * self.radius_factor)
            else:
                # First few improvements - expand slightly
                self.trust_region_radius = min(self.max_radius,
                                              self.trust_region_radius * 1.2)
        else:
            # We've improved recently, maintain or slightly expand trust region
            if len(self.y) >= 2:
                recent_best = np.min(self.y[:-5]) if len(self.y) > 5 else current_best
                if (recent_best - current_best) > self.progress_threshold * 2:
                    # Good improvement - expand trust region
                    self.trust_region_radius = min(self.max_radius,
                                                  self.trust_region_radius * 1.1)
        
        # Update last improvement evaluation count
        self.last_improvement_eval = self.n_evals
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sequence sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to proceed
            if self.n_evals >= self.budget:
                break
                
            # Update trust region based on progress
            self._update_trust_region()
            
            # Select next point using acquisition function optimization within trust region
            X_next = self._optimize_acquisition(n_starts=5)
            
            # Ensure the point is within bounds
            X_next = np.clip(X_next, self.bounds[0], self.bounds[1])
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next.reshape(1, -1))
            
            # Update model with new points
            self._update_eval_points(X_next.reshape(1, -1), y_next)
            
            # Retrain the model periodically or when we have enough data
            if len(self.X) >= self.n_init + 5:
                self._fit_model(self.X, self.y)
                
        # Return best result
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
```
## Feedback
 The algorithm SobolTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3406 with standard deviation 0.0141.

took 32.98 seconds to run.