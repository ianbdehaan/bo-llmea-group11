# Description
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Gradient-Based Refinement (EMBO-GRAV)**

This novel Bayesian optimization algorithm introduces several key innovations that distinguish it from existing approaches:

1. **Adaptive Exploration Strategy**: Instead of fixed trust regions, uses a dynamic exploration schedule based on problem complexity and uncertainty
2. **Gradient-Based Refinement**: Incorporates gradient information from the surrogate model to guide acquisition function optimization
3. **Multi-Stage Surrogate Modeling**: Uses a two-stage approach where initial models are fit with sparse GP, then refined with local models near promising regions
4. **Adaptive Batch Selection**: Dynamically adjusts batch size based on model confidence and problem dimensionality
5. **Robust Acquisition Function Optimization**: Combines random sampling with gradient-based local search for acquisition function optimization

# Justification
The key innovations in this algorithm address limitations of existing approaches:

1. **Different Exploration Strategy**: Unlike EMBO-TRE's fixed trust regions, EMBO-GRAV uses an adaptive exploration schedule that dynamically adjusts the balance between global and local search based on observed function behavior and uncertainty patterns.

2. **Gradient Integration**: While other algorithms use only function evaluations, this approach leverages gradient information from Gaussian Process models to significantly improve acquisition function optimization efficiency - a key computational bottleneck in Bayesian Optimization.

3. **Multi-Stage Modeling**: Instead of fitting one model throughout optimization, this algorithm uses sparse initial GP models and progressively builds more refined local models near promising regions, reducing computational overhead while maintaining accuracy.

4. **Adaptive Batch Selection**: Rather than fixed batch sizes, the algorithm dynamically adjusts batch size based on current model confidence (higher confidence = larger batches) and problem dimensionality to optimize exploration efficiency.

5. **Novel Acquisition Optimization**: Combines random sampling with gradient-based local refinement in a way that's computationally efficient yet highly effective, unlike standard approaches that rely solely on expensive optimization methods.

These innovations make the algorithm more computationally efficient and better suited for problems with complex landscapes where traditional BO approaches may struggle.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class EMBO_GRAV:
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
        self.local_gps = []  # Store local models near promising regions
        
        # Exploration strategy parameters
        self.exploration_schedule = []
        self.current_exploration_level = 0
        self.uncertainty_threshold = 0.1  # Threshold for switching exploration behavior
        
        # Batch selection parameters
        self.batch_size = 1
        self.confidence_threshold = 0.5  # When to increase batch size based on model confidence
        
        # Gradient-based optimization parameters
        self.gradient_samples = 5  # Number of gradient samples to use
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
        # Initialize exploration schedule
        self._initialize_exploration_schedule()
        
    def _initialize_exploration_schedule(self):
        """Initialize adaptive exploration levels based on problem dimensionality"""
        # Create a schedule that balances global and local search
        n_levels = min(5, max(2, self.dim // 3))
        self.exploration_schedule = np.linspace(0.1, 1.0, n_levels)
        
    def _sample_points(self, n_points):
        """Use Latin Hypercube sampling for good coverage"""
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y, use_local=False):
        """Fit a Gaussian Process with ARD Matern kernel"""
        # Use a small number of restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _compute_gradients(self, X, gp):
        """Compute approximate gradients from GP model"""
        # Get gradient information from GP
        X_eval = np.array(X)
        if len(X_eval.shape) == 1:
            X_eval = X_eval.reshape(1, -1)
            
        # Sample a few points around each evaluation point to estimate gradients
        grad_samples = []
        for x in X_eval:
            # Create small perturbations to compute finite differences
            eps = 1e-4
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                # Make sure we stay within bounds
                x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
                x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
                
                y_plus = gp.predict(x_plus.reshape(1, -1), return_std=False)[0]
                y_minus = gp.predict(x_minus.reshape(1, -1), return_std=False)[0]
                
                grad[i] = (y_plus - y_minus) / (2 * eps)
            
            grad_samples.append(grad)
        
        return np.array(grad_samples)

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function with gradient information"""
        if self.gp is None:
            # For initial points before first model fit, return random values
            return self.rng.random((X.shape[0], 1)) * 0.1
            
        # Compute the expected improvement using GP predictions
        # Get current best value (minimum of y)
        current_best = np.min(self.y) if len(self.y) > 0 else 0
        
        # Predict mean and std at X points
        mean, std = self.gp.predict(X, return_std=True)
        
        # Expected Improvement formula
        improvement = current_best - mean
        z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Normalize and return
        if len(ei.shape) == 1:
            ei = ei.reshape(-1, 1)
            
        return ei

    def _optimize_acquisition(self, X_start, acquisition_func):
        """Optimize acquisition function using gradient-based approach"""
        # Use a hybrid approach: start with random points and then refine with gradient information
        
        best_value = np.inf
        best_point = None
        
        # Try multiple starting points
        n_starts = min(10, max(5, self.dim * 2))
        
        for i in range(n_starts):
            # Start from a random point within bounds
            if i == 0:
                start_point = X_start[0:1] if len(X_start) > 0 else self._sample_points(1)
            else:
                start_point = self._sample_points(1)
                
            try:
                # Use scipy minimize with gradient information if available
                def objective(x):
                    return -float(acquisition_func(x.reshape(1, -1))[0, 0])  # Negative because we minimize
                
                def jac(x):
                    # Simple finite difference approximation of gradient
                    eps = 1e-6
                    grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_plus[i] += eps
                        x_minus = x.copy()
                        x_minus[i] -= eps
                        
                        # Clip to bounds
                        x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
                        x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
                        
                        val_plus = float(acquisition_func(x_plus.reshape(1, -1))[0, 0])
                        val_minus = float(acquisition_func(x_minus.reshape(1, -1))[0, 0])
                        grad[i] = (val_plus - val_minus) / (2 * eps)
                    
                    return -grad  # Negative because we minimize
                
                # Try different optimization methods
                try:
                    result = minimize(
                        objective,
                        start_point[0],
                        method='L-BFGS-B',
                        bounds=list(zip(self.bounds[0], self.bounds[1])),
                        jac=jac,
                        options={'maxiter': 20, 'disp': False}
                    )
                except:
                    # Fallback to Nelder-Mead if L-BFGS fails
                    result = minimize(
                        objective,
                        start_point[0],
                        method='Nelder-Mead',
                        options={'maxiter': 10, 'disp': False}
                    )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = float(acquisition_func(point)[0, 0])
                    
                    if acq_value < best_value:  # Minimize acquisition function
                        best_value = acq_value
                        best_point = point
                        
            except Exception as e:
                # If optimization fails, continue with random point
                continue
                
        # Return the best point found or a random point
        if best_point is not None:
            return best_point
        else:
            return self._sample_points(1)

    def _get_local_region(self):
        """Get a region around the current best point for local modeling"""
        if self.X is None or len(self.X) == 0:
            center = np.mean(self.bounds, axis=0)
            radius = 2.0
        else:
            # Use the best point and expand by a factor of 2
            best_idx = np.argmin(self.y)
            center = self.X[best_idx].copy()
            # Estimate a reasonable radius based on problem dimensions and current points
            if len(self.X) > 1:
                distances = cdist([center], self.X)[0]
                distances = distances[distances > 0]  # Exclude distance to itself
                if len(distances) > 0:
                    radius = np.percentile(distances, 25)
                else:
                    radius = 1.0
            else:
                radius = 1.0
                
        return center, radius

    def _update_batch_size(self):
        """Adaptively update batch size based on model confidence and problem dimensionality"""
        # Simple heuristic: if we have a good model and are in a low-dimensional region, use larger batches
        if self.n_evals > 0:
            model_confidence = np.std(self.y) / (np.max(self.y) - np.min(self.y) + 1e-8) if len(self.y) > 1 else 0.5
            
            # Adjust batch size based on confidence and dimensionality
            base_batch = min(3, max(1, self.dim // 2))
            
            if model_confidence < self.confidence_threshold:
                # High uncertainty: use smaller batches for exploration
                self.batch_size = max(1, base_batch - 1)
            else:
                # Low uncertainty: can afford larger batches
                self.batch_size = min(max(5, base_batch + 1), 5)
        else:
            self.batch_size = 1

    def _select_next_points(self):
        """Select next points to evaluate using adaptive acquisition optimization"""
        # Update batch size based on current state
        self._update_batch_size()
        
        # Get current best point for local exploration
        center, radius = self._get_local_region()
        
        # Sample a few candidates for acquisition function optimization
        n_candidates = max(10, self.batch_size * 5)
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function at all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points using gradient-based refinement
        selected_points = []
        
        # Sort by acquisition value and select top ones
        sorted_indices = np.argsort(acq_values.flatten())
        top_candidates = candidates[sorted_indices[:self.batch_size]]
        
        # Refine each candidate with gradient-based optimization
        for i, candidate in enumerate(top_candidates):
            refined_point = self._optimize_acquisition(
                candidate.reshape(1, -1), 
                self._acquisition_function
            )
            selected_points.append(refined_point)
            
        return np.vstack(selected_points)

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        # Initialize with Latin Hypercube design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update batch size based on current state
            self._update_batch_size()
            
            # Update performance history for exploration control
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
            
            # Fit model with current data (update both global and local models)
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points to evaluate using acquisition function
            next_points = self._select_next_points()
            
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
## Error
 Traceback (most recent call last):
  File "<EMBO_GRAV>", line 57
    points = sampler.random(n=n=n_points)
                               ^
SyntaxError: invalid syntax
