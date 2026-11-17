from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_TRE_TS:
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
        self.tr_size = 1.0  # Initial trust region size
        self.tr_center = None  # Current trust region center
        self.tr_growth_factor = 1.2  # Growth factor for trust region
        self.tr_shrink_factor = 0.5  # Shrink factor for trust region
        self.tr_min_size = 0.01  # Minimum trust region size
        
        # Acquisition function parameters
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Sobol sampling for better coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a sparse Gaussian Process with ARD Matern kernel
        # Use a small number of hyperparameter restarts for efficiency
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Limited restarts for efficiency
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y.flatten())
        return gp

    def _acquisition_function(self, X):
        # Thompson sampling from GP posterior
        if self.gp is None:
            # For initial points before first model fit, return a small value to encourage exploration
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP (mean and std)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Thompson sampling: sample from posterior distribution
        # Draw samples from the GP posterior for each point
        samples = self.rng.normal(mean, std)
        
        # Return negative of samples for maximization (minimization context)
        return -samples.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization with trust region"""
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
            # For larger batches, use a combination approach with trust region
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points within trust region around the best point
            if len(selected_points) > 0 and self.tr_center is not None:
                # Sample from a region around the current best point to maintain diversity
                center = selected_points[0][0]  # Use first point as center for trust region
                
                # Define trust region bounds based on current center and size
                tr_lower = np.maximum(self.bounds[0], center - self.tr_size)
                tr_upper = np.minimum(self.bounds[1], center + self.tr_size)
                
                # Generate additional points within trust region
                remaining = batch_size - len(selected_points)
                if remaining > 0:
                    for _ in range(remaining):
                        # Sample within trust region bounds
                        point = tr_lower + self.rng.random(self.dim) * (tr_upper - tr_lower)
                        selected_points.append(point.reshape(1, -1))
            else:
                # Fallback to random sampling if no center available
                remaining = batch_size - len(selected_points)
                for _ in range(remaining):
                    selected_points.append(self._sample_points(1))
            
            return np.vstack(selected_points)

    def _update_trust_region(self, new_point, improvement):
        """Update trust region based on recent progress"""
        if self.tr_center is None:
            # Initialize trust region center
            self.tr_center = new_point.copy()
            return
            
        # Update center with new best point if it's an improvement
        if improvement > 0:
            self.tr_center = new_point.copy()
            
            # Shrink trust region when progress is made
            self.tr_size = max(self.tr_min_size, self.tr_size * self.tr_shrink_factor)
        else:
            # Grow trust region when no significant progress is made
            self.tr_size = min(5.0, self.tr_size * self.tr_growth_factor)

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
        
        # Initialize with Sobol design for good coverage
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # Update trust region based on the best point found so far
            if len(self.y) > 0:
                current_best_idx = np.argmin(self.y)
                current_best_point = self.X[current_best_idx]
                current_best_value = float(self.y[current_best_idx])
                
                # For trust region update, we need to determine improvement
                if self.n_evals > self.n_init:  # Only after initial points
                    # Calculate improvement compared to previous best (simple approach)
                    prev_best_y = np.min(self.y[:-batch_size]) if len(self.y) > batch_size else current_best_value
                    improvement = prev_best_y - current_best_value
                    
                    # Update trust region based on progress
                    self._update_trust_region(current_best_point, improvement)
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
