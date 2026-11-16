from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
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
        self.radius_growth_factor = 1.5
        self.radius_shrink_factor = 0.7
        self.progress_window = 10
        self.last_improvements = []
        
        # Model and acquisition function
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
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
        # Implement Expected Improvement acquisition function
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
        best_acq_val = -np.inf
        best_point = None
        
        # Sample starting points
        start_points = self._sample_points(n_starts)
        
        for i in range(n_starts):
            # Add small perturbation to avoid local minima
            x0 = start_points[i] + (self.rng.random(self.dim) - 0.5) * 0.1
            
            # Optimize acquisition function with L-BFGS-B
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0],
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 100}
                )
                
                if result.success:
                    acq_val = -result.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x
            except:
                # If optimization fails, continue with next start point
                continue
                
        return best_point
    
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
            
            # Track improvements for progress monitoring
            self.last_improvements.append(self.y[best_idx])
            if len(self.last_improvements) > self.progress_window:
                self.last_improvements.pop(0)
    
    def _adapt_trust_region(self):
        # Adapt trust region based on optimization progress
        
        # If we have enough data to analyze progress
        if len(self.last_improvements) >= 2:
            # Check if there has been improvement in recent iterations
            recent_improvements = self.last_improvements[-min(self.progress_window, len(self.last_improvements)):]
            
            # Calculate the relative change
            if len(recent_improvements) > 1:
                improvement_rate = (recent_improvements[0] - recent_improvements[-1]) / (recent_improvements[0] + 1e-8)
                
                # If there has been little improvement, shrink or restart trust region
                if improvement_rate < 0.01:  # 1% improvement threshold
                    # Shrink trust region if it's not too small
                    if self.trust_region_radius > self.min_radius:
                        self.trust_region_radius *= self.radius_shrink_factor
                        
                    # If trust region becomes too small, restart with larger radius
                    if self.trust_region_radius <= self.min_radius * 2:
                        self.trust_region_radius = self.max_radius / 2.0
                else:
                    # If there's significant improvement, expand trust region
                    if self.trust_region_radius < self.max_radius:
                        self.trust_region_radius *= self.radius_growth_factor
                        
                    # Ensure radius doesn't exceed maximum
                    self.trust_region_radius = min(self.trust_region_radius, self.max_radius)
            else:
                # If not enough data points, maintain current radius
                pass
        else:
            # For early iterations, maintain default radius
            pass
    
    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using trust region strategy
        
        # First, try to find a good point using acquisition function optimization within trust region
        if self.trust_region_center is not None:
            # Generate candidates in the current trust region
            candidates = []
            
            # Sample points around trust region center
            region_samples = self._sample_points(20)
            
            # Filter by trust region
            distances = np.linalg.norm(region_samples - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            
            if len(valid_indices) > 0:
                candidates.extend(region_samples[valid_indices])
            
            # If we don't have enough candidates from trust region, add more
            if len(candidates) < 10:
                additional_samples = self._sample_points(10 - len(candidates))
                candidates.extend(additional_samples)
            
            # Also add some points near best known point for local refinement
            if len(self.X) > 0:
                best_idx = np.argmin(self.y)
                best_point = self.X[best_idx]
                
                # Add some perturbations around best point
                for _ in range(5):
                    perturbation = (self.rng.random(self.dim) - 0.5) * self.trust_region_radius * 0.2
                    candidate = best_point + perturbation
                    # Ensure it's within bounds
                    candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                    candidates.append(candidate)
            
            candidates = np.array(candidates[:30])  # Limit candidates
            
            # Optimize acquisition function to find best candidate
            acq_values = self._acquisition_function(candidates)
            best_idx = np.argmax(acq_values)
            
            return candidates[best_idx:best_idx+1]
        else:
            # For initial phase, use random sampling
            return self._sample_points(1)
    
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
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to proceed
            if self.n_evals >= self.budget:
                break
                
            # Adapt trust region based on progress
            self._adapt_trust_region()
            
            # Select next points to evaluate (batch size = 1 for simplicity)
            batch_size = min(1, self.budget - self.n_evals)
            X_next = self._select_next_points(batch_size)
            
            # Evaluate next points
            y_next = self._evaluate_points(func, X_next)
            
            # Update model with new points
            self._update_eval_points(X_next, y_next)
            
            # Retrain the model periodically or when we have enough data
            if len(self.X) >= self.n_init + 5:
                self._fit_model(self.X, self.y)
                
        # Return best result
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
