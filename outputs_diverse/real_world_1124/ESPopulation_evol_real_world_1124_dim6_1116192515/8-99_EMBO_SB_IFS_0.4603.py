from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EMBO_SB_IFS:
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
        
        # Acquisition function parameters
        # Dynamic exploration weight based on progress through budget
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest regressor with better uncertainty estimation
        # Use bootstrap sampling for variance estimates
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # No parallel processing to maintain reproducibility
        )
        
        rf.fit(X, y.flatten())
        return rf

    def _get_rf_uncertainty(self, X):
        """Estimate uncertainty from Random Forest predictions using bootstrap approach"""
        if self.rf is None:
            return np.ones((X.shape[0], 1)) * 1.0
            
        # Get predictions for all points
        predictions = []
        n_trees = self.rf.n_estimators
        
        # Collect predictions from each tree (simplified approach)
        # We approximate uncertainty using the spread of predictions across trees
        # For a more accurate approach, we would need to implement custom forest with variance estimates
        
        # Get mean prediction
        mean_pred = self.rf.predict(X)
        
        # For simplicity, we estimate uncertainty as:
        # 1. Standard deviation of training targets if available
        # 2. Or use a fixed factor times the range of function values
        if self.y is not None and len(self.y) > 0:
            std_train = np.std(self.y)
            if std_train == 0:
                std_train = 1.0
        else:
            std_train = 1.0
            
        # Use a dynamic uncertainty based on function value range
        if self.y is not None and len(self.y) > 1:
            range_y = np.max(self.y) - np.min(self.y)
            if range_y == 0:
                range_y = 1.0
            uncertainty = range_y * 0.1  # Proportional to function range
        else:
            uncertainty = std_train
            
        # Return uncertainty estimate (we'll use a constant for now but could be improved)
        return np.full((X.shape[0], 1), uncertainty)

    def _acquisition_function(self, X):
        # Lower Confidence Bound with adaptive exploration weight using RF predictions
        if self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from RF (mean)
        mean = self.rf.predict(X)
        
        # Get uncertainty estimate
        uncertainty = self._get_rf_uncertainty(X)
        
        # Determine current exploration weight based on progress through budget
        progress = self.n_evals / self.budget
        if len(self.acq_weight_schedule) > 0:
            # Interpolate between weights based on progress
            if progress >= 1.0:
                acq_weight = self.acq_weight_schedule[-1]
            else:
                idx = int(progress * (len(self.acq_weight_schedule) - 1))
                idx = min(idx, len(self.acq_weight_schedule) - 1)
                acq_weight = self.acq_weight_schedule[idx]
        else:
            acq_weight = 1.0
            
        # LCB acquisition function using mean and uncertainty estimate
        lcb = mean - acq_weight * uncertainty.flatten()
        
        return lcb.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=30):
        """Optimize acquisition function using multiple starting points with better strategies"""
        # Generate diverse starting points - use a mix of Sobol and random sampling
        start_points = self._sample_points(n_starts // 2)
        
        # Add some random points for diversity
        random_points = self._sample_points(n_starts // 2)
        start_points = np.vstack([start_points, random_points])
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception as e:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization"""
        # For small batches, use a more focused approach
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use multi-start L-BFGS to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=15)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach with better diversity
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=25)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using random sampling with some structure
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample from a region around the current best point to maintain diversity
                if self.X is not None and len(self.X) > 0:
                    # Use recent best points to define search region
                    recent_best_idx = np.argsort(self.y.flatten())[:min(5, len(self.y))]
                    recent_best_points = self.X[recent_best_idx]
                    # Add some variation by sampling from a local region around the best point
                    if len(recent_best_points) > 0:
                        center = recent_best_points[np.argmin(self.y[recent_best_idx])]
                        # Sample around the center point with some randomness
                        for _ in range(remaining):
                            # Use a different randomization approach for diversity
                            sample_point = center + self.rng.normal(0, 0.5, self.dim)
                            # Clip to bounds
                            sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                            selected_points.append(sample_point.reshape(1, -1))
                    else:
                        # Fallback to random sampling
                        for _ in range(remaining):
                            selected_points.append(self._sample_points(1))
                else:
                    # Fallback to random sampling
                    for _ in range(remaining):
                        selected_points.append(self._sample_points(1))
            
            return np.vstack(selected_points)

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
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.rf = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            elif remaining_budget <= 20:
                batch_size = min(4, remaining_budget)
            else:
                # For larger budgets, use a more sophisticated adaptive approach
                # Decrease batch size as we get closer to budget limit
                base_batch = max(1, min(5, remaining_budget // 8))
                batch_size = max(1, base_batch - int((self.budget - remaining_budget) / (self.budget * 0.3)))
                batch_size = min(batch_size, remaining_budget)
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.rf = self._fit_model(self.X, self.y)
            
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
