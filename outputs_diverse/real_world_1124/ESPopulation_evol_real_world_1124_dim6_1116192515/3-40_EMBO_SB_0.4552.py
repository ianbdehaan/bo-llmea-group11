from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_SB:
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
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.region_shrink_factor = 0.8
        self.region_expand_factor = 1.2
        self.progress_threshold = 1e-3
        self.last_improvement_evals = 0
        self.improvement_count = 0
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
        # Batch selection parameters
        self.batch_size = 1
        self.initial_batch_size = min(5, max(1, budget // 20))
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use fewer restarts initially and more as we get more data to balance efficiency and quality
        n_restarts = max(3, min(10, self.n_evals // 5))
        
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        # Handle numerical issues during fitting
        try:
            gp.fit(X, y)
        except Exception:
            # If fitting fails, use a simpler model
            warnings.warn("GP fitting failed, using simplified model")
            kernel_simple = C(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel_simple,
                n_restarts_optimizer=max(1, n_restarts//2),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            gp.fit(X, y)
            
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current best value (with some smoothing to avoid numerical issues)
        if len(self.y) > 0:
            current_best = np.min(self.y) - 1e-8
        else:
            current_best = 0.0
            
        # Compute EI
        improvement = current_best - mean
        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _get_trust_region_center(self):
        # Return the best point seen so far as the center of trust region
        if self.X is None or len(self.X) == 0:
            return np.mean(self.bounds, axis=0)
        else:
            best_idx = np.argmin(self.y)
            return self.X[best_idx].copy()

    def _update_trust_region(self):
        # Update trust region based on recent progress
        if len(self.performance_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        recent_improvement = (self.performance_history[-1] - self.performance_history[-2]) / abs(self.performance_history[-1])
        
        if recent_improvement > self.progress_threshold:
            # We're making good progress, expand the trust region
            self.trust_region_radius = min(
                self.max_trust_region_radius,
                self.trust_region_radius * self.region_expand_factor
            )
            self.improvement_count += 1
        else:
            # No significant improvement, shrink the trust region
            self.trust_region_radius = max(
                self.min_trust_region_radius,
                self.trust_region_radius * self.region_shrink_factor
            )
            self.improvement_count = 0

    def _sample_in_trust_region(self, center, n_points):
        # Sample points within trust region around the center
        # Create a box around center with size based on current trust region radius
        lower_bound = np.maximum(self.bounds[0], center - self.trust_region_radius)
        upper_bound = np.minimum(self.bounds[1], center + self.trust_region_radius)
        
        # If region is too small, just sample from bounds
        if np.all(upper_bound - lower_bound > 1e-6):
            # Use Latin Hypercube sampling within the trust region
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            points = sampler.random(n=n_points)
            points = lower_bound + points * (upper_bound - lower_bound)
        else:
            # Fall back to random sampling when trust region is very small
            points = self._sample_points(n_points)
            
        return points

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        # Use L-BFGS with multiple restarts for better efficiency than differential evolution
        
        n_starts = min(30, max(10, batch_size * 5))  # More starts for better diversity
        
        # Get current trust region center and update it
        tr_center = self._get_trust_region_center()
        
        # Generate diverse starting points within trust region
        start_points = self._sample_in_trust_region(tr_center, n_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use L-BFGS optimization around this point to improve it
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20},
                    tol=1e-6
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value > best_value:  # Maximize EI
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from trust region
            selected_points = self._sample_in_trust_region(tr_center, 1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points within trust region
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_in_trust_region(tr_center, 1)
                candidates.append(rand_point)
                
        return np.vstack(candidates)[:batch_size]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update trust region based on recent performance
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
                
                # Update trust region every few iterations or when we have enough data
                if (self.n_evals - self.last_improvement_evals > max(10, self.budget // 20)) and len(self.performance_history) >= 2:
                    self._update_trust_region()
                    self.last_improvement_evals = self.n_evals
            
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 10:
                self.batch_size = 1
            elif remaining_budget <= 50:
                self.batch_size = min(3, remaining_budget)
            else:
                self.batch_size = self.initial_batch_size
            
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(self.batch_size)
            
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
