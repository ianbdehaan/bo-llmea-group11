from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

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
        self.n_init = min(max(10, 2 * dim + 1), max(10, budget // 4))
        if self.n_init > budget:
            self.n_init = budget
            
        # Acquisition function parameters
        self.noise_level = 1e-6
        
        # Model parameters
        self.model = None
        
        # Trust region parameters
        self.tr_radius = 5.0  # Initial trust region radius (full search space)
        self.tr_center = None  # Center of current trust region
        self.min_tr_radius = 0.01  # Minimum trust region radius
        self.tr_shrink_factor = 0.5  # Factor to shrink when progress stalls
        self.tr_expand_factor = 2.0  # Factor to expand when progress is good
        self.progress_window = max(5, dim)  # Number of recent evaluations to consider for progress
        self.last_improvements = []  # Track improvements over recent evaluations
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model with reduced computational overhead"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with sparse approximation to reduce computational cost
        # Use fewer restarts for efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced from 3 to 2 for speed
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function with adaptive exploration"""
        # If no model exists, return zeros
        if self.model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from the GP model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Calculate EI using standard formula with adaptive exploration weight
        progress_ratio = self.n_evals / self.budget
        
        # More aggressive decrease in exploration for later iterations
        xi = 0.01 * (1 - progress_ratio**2) + 0.001
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = y_min - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using a hybrid approach within trust region"""
        # Sample points within the current trust region
        candidates = self._sample_points(n_starting_points)
        
        # Filter candidates to be within trust region
        if self.tr_center is not None:
            distances = cdist(candidates, self.tr_center.reshape(1, -1))
            valid_indices = np.where(distances.flatten() <= self.tr_radius)[0]
            candidates = candidates[valid_indices]
            
            # If too few candidates remain, expand trust region
            if len(candidates) < 5:
                self.tr_radius = min(self.tr_radius * 2.0, 5.0)
                distances = cdist(candidates, self.tr_center.reshape(1, -1))
                valid_indices = np.where(distances.flatten() <= self.tr_radius)[0]
                candidates = candidates[valid_indices]
        
        # If no valid candidates after filtering, sample new ones
        if len(candidates) == 0:
            candidates = self._sample_points(n_starting_points)
            
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points to start optimization from
        top_indices = np.argsort(acq_values.flatten())[-max(5, self.dim):]
        starting_points = candidates[top_indices]
        
        # Use local optimization around the best candidates
        best_acq_val = -np.inf
        best_point = None
        
        for x0 in starting_points:
            try:
                # Simple bounded L-BFGS optimization with restarts
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    acq_val = -result.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x
                        
            except:
                # If optimization fails, continue with current best
                pass
        
        # Fallback to the best candidate from sampling if optimization fails
        if best_point is None:
            best_idx = np.argmax(acq_values.flatten())
            best_point = candidates[best_idx]
            
        return best_point.reshape(1, -1)

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

    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        # Track recent improvements
        if len(self.y) >= 2:
            recent_improvements = []
            for i in range(1, min(len(self.y), self.progress_window)):
                improvement = self.y[-i-1] - self.y[-i]
                recent_improvements.append(improvement)
            
            if len(recent_improvements) > 0:
                avg_improvement = np.mean(recent_improvements)
                
                # Update trust region based on progress
                if avg_improvement < 0.01:  # No significant improvement
                    # Shrink trust region
                    self.tr_radius = max(self.min_tr_radius, self.tr_radius * self.tr_shrink_factor)
                else:  # Significant improvement
                    # Expand trust region
                    self.tr_radius = min(5.0, self.tr_radius * self.tr_expand_factor)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Set trust region center to best point found so far
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Update trust region based on recent progress
            self._update_trust_region()
            
            # Select next points using acquisition function optimization within trust region
            next_points = self._optimize_acquisition(n_starting_points=min(50, max(10, 2 * self.dim)))
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update trust region center to the best point found so far
            best_idx = np.argmin(self.y)
            self.tr_center = self.X[best_idx].copy()
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
