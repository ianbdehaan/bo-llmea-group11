from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings

class AdaptiveTrustRegionBOImproved:
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
        
        # Model parameters - using ensemble of models for better uncertainty quantification
        self.models = None
        self.scaler_X = None
        self.scaler_y = None
        
        # Acquisition function parameters
        self.acq_type = 'ei_thompson'  # Switch between EI and EI+Thompson sampling
        self.thompson_samples = 5  # Number of Thompson samples for hybrid acquisition
        
        # Convergence tracking
        self.convergence_history = []
        self.exploitation_phase = False  # Whether we're in exploitation phase
        
    def _sample_points(self, n_points):
        # Use Latin hypercube sampling for diverse initial points
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_models(self, X, y):
        """Fit ensemble of Gaussian Processes with different kernels"""
        # Scale the data for better GP performance
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Define multiple kernels for ensemble
        kernels = [
            C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5),
            C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        ]
        
        models = []
        for kernel in kernels:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=2,  # Fewer restarts to save computation
                normalize_y=False,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X_scaled, y_scaled)
            models.append(gp)
        
        # Store scalers and models for later use
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.models = models
        
        return models
    
    def _acquisition_function(self, X):
        """Compute acquisition function value for points in X"""
        if self.models is None:
            # If no model, return random values
            return self.rng.random(len(X))
        
        # Transform X to scaled space
        X_scaled = self.scaler_X.transform(X)
        
        # Get predictions from all models and compute ensemble mean/std
        means = []
        stds = []
        for model in self.models:
            mean, std = model.predict(X_scaled, return_std=True)
            means.append(mean)
            stds.append(std)
        
        # Compute ensemble statistics
        means = np.array(means)  # (n_models, n_points)
        stds = np.array(stds)
        
        mean_pred = np.mean(means, axis=0)
        std_pred = np.sqrt(np.mean(stds**2, axis=0) + np.var(means, axis=0))
        
        # Get best observed value (from original scale)
        y_best = np.min(self.y) if len(self.y) > 0 else 0.0
        
        # Transform y_best to scaled space
        y_best_scaled = self.scaler_y.transform([[y_best]]).ravel()[0]
        
        if self.acq_type == 'ei_thompson':
            # Combine Expected Improvement with Thompson sampling
            improvement = y_best_scaled - mean_pred
            z = improvement / (std_pred + 1e-8)
            ei = improvement * norm.cdf(z) + std_pred * norm.pdf(z)
            
            # Add Thompson sampling component
            thompson_samples = []
            for _ in range(self.thompson_samples):
                samples = []
                for model in self.models:
                    sample = model.sample_y(X_scaled, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
                    samples.append(sample.flatten())
                samples = np.array(samples)
                thompson_mean = np.mean(samples, axis=0)
                thompson_samples.append(thompson_mean)
            
            thompson_samples = np.array(thompson_samples)
            thompson_improvement = y_best_scaled - np.mean(thompson_samples, axis=0)
            thompson_improvement = np.maximum(thompson_improvement, 0)
            
            # Combine both acquisition components
            combined_acq = ei + 0.5 * thompson_improvement
            
            return combined_acq
        else:
            # Standard Expected Improvement
            improvement = y_best_scaled - mean_pred
            z = improvement / (std_pred + 1e-8)
            ei = improvement * norm.cdf(z) + std_pred * norm.pdf(z)
            return ei
    
    def _optimize_acquisition(self, n_starts=10):
        """Optimize acquisition function using hybrid approach"""
        if self.models is None:
            # Return random point if no model yet
            return self._sample_points(1)[0]
        
        best_acq_value = -np.inf
        best_point = None
        
        # Sample initial points for multi-start optimization
        initial_points = self._sample_points(n_starts)
        
        # Add trust region center as an additional starting point if available
        if self.trust_region_center is not None:
            initial_points = np.vstack([initial_points, self.trust_region_center])
            
        # Optimization approach: Use a small evolution strategy with local refinement
        # First sample candidates in trust region
        candidates = self._sample_points(50)
        
        # Filter candidates within current trust region
        if self.trust_region_center is not None:
            distances = np.linalg.norm(candidates - self.trust_region_center, axis=1)
            candidates = candidates[distances <= self.trust_region_radius]
        
        # If no candidates in trust region, use full space sampling
        if len(candidates) == 0:
            candidates = self._sample_points(50)
            
        acq_values = self._acquisition_function(candidates)
        
        # Select best candidate from this batch
        best_idx = np.argmax(acq_values)
        if acq_values[best_idx] > best_acq_value:
            best_acq_value = acq_values[best_idx]
            best_point = candidates[best_idx].copy()
            
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
        """Enhanced trust region adaptation"""
        if self.n_evals <= self.n_init:
            return  # Don't modify trust region during initial phase
            
        # Track convergence
        if len(self.y) >= 2:
            current_best = np.min(self.y)
            recent_improvement = False
            
            # Check for significant improvement in last few evaluations
            if len(self.y) >= 10:
                recent_best = np.min(self.y[-10:])
                improvement = recent_best - current_best
                if improvement > self.progress_threshold * 2:
                    recent_improvement = True
                    
            # Update exploitation phase flag based on convergence
            if not recent_improvement and len(self.y) > 20:
                # If we haven't improved much recently, switch to exploitation
                self.exploitation_phase = True
            elif recent_improvement and self.exploitation_phase:
                # If we're improving again, we can be more exploratory
                self.exploitation_phase = False
            
        # Adaptive trust region based on phase and improvement
        if self.exploitation_phase:
            # In exploitation phase, reduce trust region size for focused search
            self.trust_region_radius = max(self.min_radius, 
                                          self.trust_region_radius * 0.8)
        else:
            # In exploration phase, allow larger radius
            if len(self.y) > 10 and len(self.y) % 5 == 0:
                # Occasionally expand for exploration
                self.trust_region_radius = min(self.max_radius,
                                              self.trust_region_radius * 1.1)
        
        # Ensure trust region stays within bounds
        self.trust_region_radius = np.clip(self.trust_region_radius, 
                                          self.min_radius, self.max_radius)
        
        # Update last improvement evaluation count
        self.last_improvement_eval = self.n_evals
    
    def _switch_acquisition_strategy(self):
        """Switch between acquisition strategies based on progress"""
        if len(self.y) < 20:
            # Early stages: use EI + Thompson for better exploration
            self.acq_type = 'ei_thompson'
        elif np.std(self.y[-10:]) < 0.1 and len(self.y) > 30:
            # Stable region: switch to pure EI for exploitation
            self.acq_type = 'ei'
        else:
            # Mixed regime: keep hybrid approach
            self.acq_type = 'ei_thompson'
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin hypercube sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial models
        self._fit_models(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to proceed
            if self.n_evals >= self.budget:
                break
                
            # Update trust region based on progress
            self._update_trust_region()
            
            # Switch acquisition strategy if needed
            self._switch_acquisition_strategy()
            
            # Select next point using acquisition function optimization within trust region
            X_next = self._optimize_acquisition(n_starts=5)
            
            # Ensure the point is within bounds
            X_next = np.clip(X_next, self.bounds[0], self.bounds[1])
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next.reshape(1, -1))
            
            # Update model with new points
            self._update_eval_points(X_next.reshape(1, -1), y_next)
            
            # Retrain the models periodically or when we have enough data
            if len(self.X) >= self.n_init + 5:
                self._fit_models(self.X, self.y)
                
        # Return best result
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
