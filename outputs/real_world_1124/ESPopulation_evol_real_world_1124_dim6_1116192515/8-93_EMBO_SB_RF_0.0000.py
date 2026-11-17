from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings

class EMBO_SB_RF:
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
        
        # Batch selection parameters
        self.batch_size_schedule = []
        self._initialize_batch_schedule()
        
        # Exploration control
        self.exploration_weight = 1.0
        self.exploration_decay = 0.95
        self.min_exploration_weight = 0.1
        
        # For tracking recent performance
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
    def _initialize_batch_schedule(self):
        # Create adaptive batch schedule that starts with larger batches and decreases
        # This allows for rapid exploration early on and fine-tuning later
        if self.budget <= 100:
            # Small budget - use fixed batch size of 1
            self.batch_size_schedule = [1] * (self.budget // 2)
        else:
            # Larger budget - start with larger batches and decrease
            max_batch_size = min(5, max(1, self.budget // 50))
            batch_sizes = []
            
            # Create schedule that starts with large batches and decreases
            remaining_budget = self.budget - self.n_init
            
            if remaining_budget > 0:
                # Determine how many batches we'll need based on the budget
                n_batches = max(1, remaining_budget // (max_batch_size * 2))
                
                # Create decreasing batch sizes
                for i in range(n_batches):
                    batch_size = max(1, max_batch_size - i * max(1, max_batch_size // (n_batches + 1)))
                    batch_sizes.extend([batch_size] * min(batch_size * 2, remaining_budget))
                    remaining_budget -= batch_size * 2
                    
                    if remaining_budget <= 0:
                        break
                        
                # Ensure we have enough batches to cover the budget
                while len(batch_sizes) < (self.budget - self.n_init):
                    batch_sizes.append(1)
                
            else:
                batch_sizes = [1] * (self.budget - self.n_init)
            
            self.batch_size_schedule = batch_sizes[:self.budget - self.n_init]
        
        # Ensure we have at least one element
        if not self.batch_size_schedule:
            self.batch_size_schedule = [1]

    def _sample_points(self, n_points):
        # Use Sobol sequence for superior space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest regressor with good uncertainty estimation
        # Use few trees for computational efficiency while maintaining good performance
        
        # Use cross-validation to estimate optimal number of trees if needed
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=False,  # OOB score computation is expensive and not necessary here
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single-threaded for reproducibility
        )
        
        rf.fit(X, y.ravel())
        return rf

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function with uncertainty adjustment
        if self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from Random Forest
        predictions = self.rf.predict(X)
        
        # Get prediction variances (uncertainty estimates)
        if hasattr(self.rf, 'oob_prediction_'):
            # Use OOB prediction variance if available
            variances = np.var(self.rf.oob_prediction_)
        else:
            # Approximate uncertainty from the forest by using variance of predictions across trees
            # For simplicity in this implementation, we'll use a fixed uncertainty value
            # In a more sophisticated version, we'd compute this properly
            variances = np.full(X.shape[0], 1e-4)
        
        # Use expected improvement formula with exploration weight
        if len(self.y) > 0:
            best_y = np.min(self.y)
            # Expected Improvement calculation for Random Forest
            # EI(x) = (mu(x) - y_best - xi) * Phi(Z) + sigma(x) * phi(Z)
            # where Z = (mu(x) - y_best - xi) / sigma(x)
            
            # Simplified version: use variance as uncertainty measure and compute a basic improvement metric
            ei_values = np.maximum(0, best_y - predictions.reshape(-1))  # Simple improvement measure
            
            # Add exploration component based on uncertainty
            exploration_component = self.exploration_weight * np.sqrt(variances)
            
            return (ei_values + exploration_component).reshape(-1, 1)
        else:
            # If no data yet, just use uncertainty as acquisition function
            return -variances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(20, max(5, batch_size * 3))  # More starts than before to ensure good coverage
        
        # Generate diverse starting points using Sobol sampling
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=15,  # Reduced iterations for speed
                    popsize=3,   # Reduced population size for speed
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:  # Minimize acquisition function
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
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
            for _ in range(min(batch_size - 1, 5)):  # Reduced diversity to save time
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
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.rf = self._fit_model(self.X, self.y)
        
        # Optimization loop
        batch_idx = 0
        
        while self.n_evals < self.budget:
            # Update exploration weight based on progress
            if len(self.performance_history) > 1 and self.exploration_weight > self.min_exploration_weight:
                improvement_ratio = (self.performance_history[-2] - self.performance_history[-1]) / abs(self.performance_history[-2])
                if improvement_ratio > 0.01:  # Significant improvement
                    self.exploration_weight *= self.exploration_decay
            
            # Determine batch size based on schedule and remaining budget
            if batch_idx < len(self.batch_size_schedule):
                batch_size = min(self.batch_size_schedule[batch_idx], self.budget - self.n_evals)
            else:
                # If we've exhausted the schedule, use a small fixed batch size
                batch_size = min(3, self.budget - self.n_evals)
            
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
            
            # Update performance tracking
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
            
            batch_idx += 1
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
