from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

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
        
        # Switch from global to local search parameters
        self.global_search_phase = True
        self.switch_threshold = max(0.3, min(0.7, (budget / 1000.0)))  # Adaptive switch point
        self.local_search_center = None
        self.local_search_radius = 5.0
        
        # For tracking performance to decide when to switch
        self.performance_history = []
        self.performance_window_size = max(5, budget // 20)
        self.last_switch_evals = 0
        
        # Batch selection parameters
        self.batch_size = 1
        self.max_batch_size = min(5, max(1, budget // 20))
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
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

    def _should_switch_to_local_search(self):
        # Check if we should switch to local search based on performance
        if len(self.performance_history) < 5:
            return False
            
        # Calculate recent improvement rate
        recent_improvement = (self.performance_history[-1] - self.performance_history[-min(5, len(self.performance_history))]) / abs(self.performance_history[-1])
        
        # Switch if we're making good progress and have enough evaluations
        if recent_improvement > 0.001 and self.n_evals > self.budget * self.switch_threshold:
            return True
            
        return False

    def _sample_local_points(self, center, n_points):
        # Sample points in a local region around the center
        if self.local_search_phase:
            # Use the current best point as center for local search
            lower_bound = np.maximum(self.bounds[0], center - self.local_search_radius)
            upper_bound = np.minimum(self.bounds[1], center + self.local_search_radius)
            
            # If region is too small, just sample from bounds
            if np.all(upper_bound - lower_bound > 1e-6):
                # Use Latin Hypercube sampling within the local region
                sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
                points = sampler.random(n=n_points)
                points = lower_bound + points * (upper_bound - lower_bound)
            else:
                # Fall back to random sampling when region is very small
                points = self._sample_points(n_points)
        else:
            # Global search - sample from entire domain
            points = self._sample_points(n_points)
            
        return points

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(20, max(5, batch_size * 3))  # More starts than needed to ensure diversity
        
        # Determine sampling approach based on search phase
        if self.global_search_phase:
            # For global search, use diverse starting points
            start_points = self._sample_points(n_starts)
        else:
            # For local search, center around current best
            if self.X is not None and len(self.X) > 0:
                tr_center = self.X[np.argmin(self.y)]
            else:
                tr_center = np.mean(self.bounds, axis=0)
                
            start_points = self._sample_local_points(tr_center, n_starts)
        
        # Evaluate acquisition function at all starting points and select best
        best_value = -np.inf
        best_point = None
        
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=20,
                    popsize=5,
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
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
            # Fallback: select from appropriate region
            if self.global_search_phase:
                selected_points = self._sample_points(1)
            else:
                if self.X is not None and len(self.X) > 0:
                    tr_center = self.X[np.argmin(self.y)]
                else:
                    tr_center = np.mean(self.bounds, axis=0)
                selected_points = self._sample_local_points(tr_center, 1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points in the appropriate region
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                if self.global_search_phase:
                    rand_point = self._sample_points(1)
                else:
                    if self.X is not None and len(self.X) > 0:
                        tr_center = self.X[np.argmin(self.y)]
                    else:
                        tr_center = np.mean(self.bounds, axis=0)
                    rand_point = self._sample_local_points(tr_center, 1)
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
            # Update performance history for switch decision
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
                
                # Check if we should switch to local search
                if self._should_switch_to_local_search():
                    self.global_search_phase = False
                    self.last_switch_evals = self.n_evals
                    
            # Select next points to evaluate using acquisition function
            batch_size = min(self.max_batch_size, self.budget - self.n_evals)  # Limit batch size
            
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
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
