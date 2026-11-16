from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
import warnings

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Set initial design size based on dimensionality
        if dim <= 3:
            self.n_init = max(5, dim + 1)
        elif dim <= 6:
            self.n_init = max(10, 2 * dim)
        else:
            self.n_init = max(15, 3 * dim)
            
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.min_trust_region = 0.1     # Minimum trust region size
        self.max_trust_region = 5.0     # Maximum trust region size
        self.trust_region_shrink_factor = 0.8  # Factor to shrink when no progress
        self.trust_region_expand_factor = 1.2  # Factor to expand when progress
        self.last_improvement_iter = 0  # Iteration when last improvement was seen
        self.improvement_threshold = 1e-3  # Minimum improvement to consider as progress
        
        # Acquisition function ensemble
        self.acq_functions = ['ei', 'lcb', 'thompson']  # Expected Improvement, Lower Confidence Bound, Thompson Sampling
        self.current_acq_idx = 0
        self.acq_performance = {acq: [] for acq in self.acq_functions}  # Track performance
        
        # For trust region management
        self.best_point = None
        self.best_value = np.inf

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process surrogate model with ARD Matern kernel
        try:
            # Try different kernel configurations to avoid numerical issues
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e2), nu=2.5)
            
            # Add WhiteKernel for noise modeling
            kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
            
            # Fit GP with robust hyperparameter optimization
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Small regularization to avoid numerical issues
                normalize_y=True,
                n_restarts_optimizer=3,  # Limit restarts for efficiency
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X, y)
            return gp
        except Exception:
            # Fallback to simpler model if complex fitting fails
            warnings.warn("GP fitting failed, using fallback")
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e2), nu=2.5)
            kernel = kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function with adaptive exploration
        if self.X is None or self.y is None:
            # If no data yet, return random values for initial exploration
            return np.random.rand(X.shape[0], 1)
            
        # Fit model on current data
        model = self._fit_model(self.X, self.y)
        
        # Get current best value
        best_y = np.min(self.y)
        
        # Compute acquisition function based on selected method
        try:
            # Get predictions from GP
            mean, std = model.predict(X, return_std=True)
            
            # Select acquisition function based on ensemble strategy
            acq_method = self.acq_functions[self.current_acq_idx]
            
            if acq_method == 'ei':
                # Expected Improvement
                improvement = best_y - mean
                z = improvement / (std + 1e-8)
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                return ei.reshape(-1, 1)
                
            elif acq_method == 'lcb':
                # Lower Confidence Bound with adaptive exploration factor
                exploit_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.9)  # Dynamic exploration
                lcb = mean - exploit_factor * std
                return (-lcb).reshape(-1, 1)  # Negative because we minimize
                
            elif acq_method == 'thompson':
                # Thompson Sampling (sample from posterior)
                samples = model.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
                return samples.reshape(-1, 1)
                
        except Exception:
            # Fallback to random acquisition if calculation fails
            return np.random.rand(X.shape[0], 1)

    def _get_trust_region_bounds(self, point):
        """Get bounds for trust region around a given point"""
        lower = np.maximum(self.bounds[0], point - self.trust_region_radius)
        upper = np.minimum(self.bounds[1], point + self.trust_region_radius)
        return list(zip(lower, upper))

    def _select_next_points(self, batch_size):
        # Select next points using trust-region constrained optimization
        # This is more robust than simple gradient-based methods
        
        # Start with the current best point for trust region center
        if self.X is not None:
            best_idx = np.argmin(self.y)
            best_point = self.X[best_idx]
        else:
            best_point = self.bounds.mean(axis=0)  # Center of search space
            
        # Generate diverse candidates within trust region
        n_candidates = max(50, batch_size * 10)
        
        # Sample points within trust region
        candidate_points = []
        for _ in range(n_candidates):
            # Sample within trust region around best point
            sample = self.rng.uniform(
                low=np.maximum(self.bounds[0], best_point - self.trust_region_radius),
                high=np.minimum(self.bounds[1], best_point + self.trust_region_radius)
            )
            candidate_points.append(sample)
        
        candidate_points = np.array(candidate_points)
        
        # Evaluate acquisition function for candidates
        acq_values = self._acquisition_function(candidate_points).flatten()
        
        # Select top candidates based on acquisition values
        top_indices = np.argsort(acq_values)[-batch_size:]
        selected_points = candidate_points[top_indices]
        
        # Refine selection using local optimization around best points
        refined_points = []
        for point in selected_points:
            try:
                # Use differential evolution to optimize acquisition locally within trust region
                def objective(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x)[0][0]
                
                # Define bounds within trust region
                bounds = self._get_trust_region_bounds(point)
                
                result = differential_evolution(objective, bounds, seed=int(self.rng.integers(2**31 - 1)), maxiter=20)
                refined_points.append(result.x)
            except:
                # If local optimization fails, keep the original point
                refined_points.append(point)
        
        return np.array(refined_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        y_values = []
        for point in X:
            y_values.append(func(point))
        
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

    def _update_trust_region(self, current_best_value):
        """Update trust region size based on progress"""
        if self.best_value == np.inf:
            # First update
            self.best_value = current_best_value
            return
            
        improvement = self.best_value - current_best_value
        if improvement > self.improvement_threshold:
            # Significant improvement, shrink trust region for more focused search
            self.trust_region_radius *= self.trust_region_shrink_factor
            self.last_improvement_iter = self.n_evals
        else:
            # No significant improvement, expand trust region to explore new areas
            if self.n_evals - self.last_improvement_iter > 5:  # Only expand after some iterations without progress
                self.trust_region_radius *= self.trust_region_expand_factor
                
        # Ensure trust region stays within bounds
        self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region, self.max_trust_region)
        
        # Update best value
        if current_best_value < self.best_value:
            self.best_value = current_best_value

    def _switch_acquisition_function(self):
        """Switch acquisition function based on recent performance"""
        # Simple strategy: switch if we haven't improved much recently
        if len(self.acq_performance[self.acq_functions[self.current_acq_idx]]) > 5:
            recent_improvements = self.acq_performance[self.acq_functions[self.current_acq_idx]][-5:]
            avg_improvement = np.mean(recent_improvements) if len(recent_improvements) > 0 else 0
            
            # If performance has been poor, switch to next acquisition function
            if avg_improvement < 1e-4:
                self.current_acq_idx = (self.current_acq_idx + 1) % len(self.acq_functions)
                
        # Reset performance tracking for all functions except current one
        for acq in self.acq_functions:
            if acq != self.acq_functions[self.current_acq_idx]:
                self.acq_performance[acq] = []

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial design
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Track best value for trust region updates
        best_idx = np.argmin(self.y)
        self.best_value = float(self.y[best_idx])
        self.best_point = self.X[best_idx].copy()
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're near the budget, evaluate remaining points
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 1:
                break
                
            # Determine batch size (limit to remaining budget)
            batch_size = min(5, remaining_budget)
            
            # Select next points using acquisition function within trust region
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Update trust region based on new best value
            current_best_idx = np.argmin(self.y)
            current_best_value = float(self.y[current_best_idx])
            self._update_trust_region(current_best_value)
            
            # Switch acquisition function if needed
            self._switch_acquisition_function()
            
            # Break if budget exhausted
            if self.n_evals >= self.budget:
                break
        
        # Return best value and point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
