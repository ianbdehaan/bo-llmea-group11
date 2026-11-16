from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler

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
        self.n_init = max(2 * self.dim + 1, 10)  # Initial points based on dimensionality
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.trust_region_center = None
        self.trust_region_radius = 2.0
        self.min_radius = 0.1
        self.max_radius = 5.0
        self.radius_shrink_factor = 0.8
        self.radius_expand_factor = 1.2
        self.progress_window = 10
        self.progress_threshold = 1e-4
        
        # Model parameters
        self.model = None
        
    def _sample_points(self, n_points):
        # Use Latin hypercube sampling for diverse initial points
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
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
    
    def _optimize_acquisition(self, n_candidates=50):
        # Multi-start optimization of acquisition function with trust region constraint
        if self.model is None:
            # If no model, return random candidates
            return self._sample_points(1)
        
        # Generate candidates in trust region
        candidates = []
        
        # Sample from trust region
        samples_in_region = self._sample_points(n_candidates)
        distances = np.linalg.norm(samples_in_region - self.trust_region_center, axis=1)
        valid_indices = np.where(distances <= self.trust_region_radius)[0]
        if len(valid_indices) > 0:
            candidates.extend(samples_in_region[valid_indices])
        
        # If not enough candidates in trust region, add some random points
        remaining = n_candidates - len(candidates)
        if remaining > 0:
            random_samples = self._sample_points(remaining)
            candidates.extend(random_samples)
            
        candidates = np.array(candidates[:n_candidates])
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Return the candidate with highest acquisition value
        best_idx = np.argmax(acq_values)
        return candidates[best_idx:best_idx+1]
    
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
        # Adaptively adjust the trust region size based on recent progress
        
        # Check if we have enough evaluations for progress tracking
        if len(self.y) < self.progress_window:
            return
            
        # Calculate recent improvement
        recent_y = self.y[-self.progress_window:]
        if len(recent_y) >= 2:
            improvement = np.abs(recent_y[0] - recent_y[-1])
            
            # If progress is minimal, shrink the trust region
            if improvement < self.progress_threshold:
                self.trust_region_radius *= self.radius_shrink_factor
            else:
                # Otherwise, expand the trust region
                self.trust_region_radius *= self.radius_expand_factor
                
        # Ensure trust region stays within bounds
        self.trust_region_radius = np.clip(self.trust_region_radius, self.min_radius, self.max_radius)
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin hypercube sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self._fit_model(self.X, self.y)
        
        # Track progress for trust region adaptation
        best_y_history = []
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to proceed
            if self.n_evals >= self.budget:
                break
                
            # Update trust region based on recent progress
            self._update_trust_region()
            
            # Select next point to evaluate using acquisition function optimization with trust region
            X_next = self._optimize_acquisition(n_candidates=10)
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next)
            
            # Update model with new points
            self._update_eval_points(X_next, y_next)
            
            # Retrain the model periodically or when we have enough data
            if len(self.X) >= self.n_init + 5:
                self._fit_model(self.X, self.y)
                
        # Return best result
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
