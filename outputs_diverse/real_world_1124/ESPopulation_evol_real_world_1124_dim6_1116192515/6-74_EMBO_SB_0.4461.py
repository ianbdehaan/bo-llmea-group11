from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
        
        # RF model parameters
        self.rf = None
        
        # For tracking recent performance
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for superior space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest regressor - much more efficient than GP
        # Use few estimators to keep computational cost low
        rf = RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single thread for reproducibility
        )
        
        rf.fit(X, y.flatten())
        return rf

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function using Random Forest surrogate
        if self.rf is None:
            # For initial points before first model fit, return random values
            return self.rng.random((X.shape[0], 1))
            
        # Get predictions and uncertainties from RF
        y_pred = self.rf.predict(X)
        
        # For EI, we need the best observed value
        if len(self.y) > 0:
            best_y = np.min(self.y)
            # Simple approximation of uncertainty (variance of predictions)
            # In practice, more sophisticated uncertainty estimates could be used
            y_std = np.std(self.rf.predict(self.X))
            
            # Expected Improvement approximation using Gaussian assumption
            if y_std > 0:
                improvement = best_y - y_pred
                z = improvement / y_std
                ei = improvement * norm.cdf(z) + y_std * norm.pdf(z)
            else:
                ei = np.zeros_like(y_pred)
        else:
            # If no observations yet, use random values
            ei = self.rng.random((X.shape[0], 1))
            
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(20, max(5, batch_size * 3))  # More starts than needed to ensure diversity
        
        # Generate diverse starting points using Sobol sequence
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Simple local refinement using grid search around starting point
            try:
                # Create a coarse grid around the starting point
                grid_size = 5
                grid_points = []
                
                # Generate points around the current start point
                for d in range(self.dim):
                    # Create a simple 1D grid along each dimension
                    dim_range = self.bounds[1][d] - self.bounds[0][d]
                    step = dim_range / (grid_size - 1)
                    if step == 0:
                        step = 1e-6
                        
                    dim_points = np.linspace(
                        max(self.bounds[0][d], start_point[0][d] - dim_range/2),
                        min(self.bounds[1][d], start_point[0][d] + dim_range/2),
                        grid_size
                    )
                    grid_points.append(dim_points)
                
                # Generate all combinations
                from itertools import product
                grid_combinations = list(product(*grid_points))
                
                # Evaluate acquisition at each point in the grid
                grid_array = np.array(grid_combinations)
                if len(grid_array) > 0:
                    acq_values = self._acquisition_function(grid_array)
                    best_idx_in_grid = np.argmin(acq_values)
                    candidate_point = grid_array[best_idx_in_grid:best_idx_in_grid+1]
                    
                    # Check if this is better than current best
                    acq_value = acq_values[best_idx_in_grid]
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = candidate_point
                        
            except Exception:
                # If optimization fails, continue with starting point
                continue
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from Sobol sequence
            selected_points = self._sample_points(1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_points(1)
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
        
        # Initialize with Sobol sequence design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.rf = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update model with current data
            self.rf = self._fit_model(self.X, self.y)
            
            # Select next points to evaluate using acquisition function
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size
            
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
