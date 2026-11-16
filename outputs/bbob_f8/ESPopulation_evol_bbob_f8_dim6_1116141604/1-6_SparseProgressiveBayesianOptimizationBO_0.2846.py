from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans

class SparseProgressiveBayesianOptimizationBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, min(20, budget // 10))  # Adaptive initialization based on budget
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model = None
        self.refinement_threshold = 5  # Number of points after which to refine model

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model using a sparse GP with Matern kernel
        
        # Use a more robust kernel for better performance on various functions
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use a sparse GP approach with limited number of restarts for efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Reduced from original to save computation
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function using improved Expected Improvement
        # with adaptive parameters based on current optimization progress
        
        if self.model is None:
            # If no model yet, use a simple exploration-based acquisition
            return np.ones((X.shape[0], 1)) * 0.01
            
        # Get model predictions
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Get the best observed value
        if len(self.y) > 0:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Calculate EI with adaptive parameter based on optimization progress
        progress_ratio = self.n_evals / self.budget
        # Dynamic exploration factor - start high, decrease as we progress
        xi = max(0.01, 0.2 * (1 - progress_ratio**0.5))  
        
        # Expected Improvement calculation with numerical stability
        with np.errstate(divide='warn'):
            imp = best_y - mu - xi
            Z = imp / (sigma + 1e-8)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        # Clip negative values to encourage exploration but not too aggressively
        ei = np.clip(ei, 0, None)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # 1. Local search from multiple starting points (exploitation)
        # 2. Global search with acquisition function (exploration)
        
        if self.X is None or len(self.X) < 2:
            # For very early iterations, use random sampling
            return self._sample_points(batch_size)
            
        selected_points = []
        
        # Use multiple restarts of local optimization for exploitation
        n_local_restarts = min(3, batch_size)  # Reduced to save computation
        
        # Start from existing points to improve convergence
        if len(self.X) > 0:
            start_indices = self.rng.choice(len(self.X), size=min(n_local_restarts, len(self.X)), replace=False)
            start_points = self.X[start_indices]
        else:
            start_points = self._sample_points(n_local_restarts)
            
        for start_point in start_points:
            # Local optimization using scipy minimize
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 15}  # Reduced iterations for efficiency
                )
                
                if result.success:
                    selected_points.append(result.x)
            except:
                # If optimization fails, just use the start point
                pass
        
        # Fill remaining points with global acquisition function maxima
        n_global = batch_size - len(selected_points)
        if n_global > 0:
            # Sample a moderate set of candidates and select top ones by acquisition value
            candidate_points = self._sample_points(500)  # Reduced from original to save computation
            acq_values = self._acquisition_function(candidate_points)
            
            # Select the top points
            top_indices = np.argsort(acq_values.flatten())[-n_global:]
            global_points = candidate_points[top_indices]
            selected_points.extend(global_points)
        
        # If we still don't have enough points, fill with random samples
        while len(selected_points) < batch_size:
            selected_points.append(self._sample_points(1)[0])
            
        return np.array(selected_points[:batch_size])

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for point in X:
            y_vals.append(func(point))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        self.n_evals += len(X)
        return y_vals
    
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
        
        # Initialize with Latin Hypercube sampling
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch size based on remaining budget and problem characteristics
            if self.budget - self.n_evals > 100:
                batch_size = min(5, max(1, (self.budget - self.n_evals) // 25))
            else:
                batch_size = 1  # Use single point for final evaluations
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically or when we have enough data for refinement
            if self.n_evals >= self.refinement_threshold and self.n_evals % 10 == 0:
                self.model = self._fit_model(self.X, self.y)
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        return best_y, best_x
