from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class EMBO_RF_DE:
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
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest regressor
        # Use a moderate number of trees for efficiency
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

    def _acquisition_function(self, X):
        # Lower Confidence Bound with adaptive exploration weight using RF predictions
        if self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from RF (mean and std)
        mean = self.rf.predict(X)
        
        # Get prediction variances (uncertainty) for acquisition function
        # For Random Forest, we can estimate uncertainty through the variance of predictions
        # Since RF doesn't directly provide variance estimates, we'll use a simple approach:
        # Use a small constant as uncertainty to maintain exploration
        uncertainty = 1.0  # Fixed uncertainty proxy - could be improved with bootstrap
        
        # LCB acquisition function using mean and uncertainty estimate
        lcb = mean - 0.5 * uncertainty  # Using fixed multiplier for simplicity
        
        return lcb.reshape(-1, 1)

    def _optimize_acquisition_de(self, n_pop=20, max_iter=30):
        """Optimize acquisition function using differential evolution"""
        # Define bounds for DE
        de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        
        def objective(x):
            # DE minimizes, but acquisition functions are maximized, so we negate
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]
        
        try:
            # Run differential evolution with multiple restarts for better exploration
            result = differential_evolution(
                objective,
                de_bounds,
                popsize=n_pop,
                maxiter=max_iter,
                seed=int(self.rng.integers(2**31 - 1)),
                strategy='best1bin',
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if result.success:
                return result.x.reshape(1, -1)
        except Exception:
            # If DE fails, fallback to random point
            pass
            
        # Fallback to random sampling
        return self._sample_points(1)

    def _select_next_points(self, batch_size):
        """Select next points using differential evolution for acquisition optimization"""
        # For small batches, use DE-based approach
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use DE to optimize acquisition function
                best_point = self._optimize_acquisition_de(n_pop=15, max_iter=20)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach
            selected_points = []
            
            # First select one point using DE acquisition optimization
            best_point = self._optimize_acquisition_de(n_pop=20, max_iter=30)
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
                            sample_point = center + self.rng.normal(0, 0.2, self.dim)
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
        
        # Initialize with maximin Latin Hypercube design scaled to problem dimensionality
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
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
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
