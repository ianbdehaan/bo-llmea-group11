from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings

class ATRBORF:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.progress_window = 5  # Number of recent evaluations to check for progress
        self.min_progress_improvement = 0.01  # Minimum improvement to consider progress
        self.restart_threshold = 3  # Number of consecutive non-improving steps before restart
        self.restart_count = 0
        self.consecutive_no_improvement = 0
        self.min_trust_region = 0.05
        self.max_trust_region = 2.0
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling

    def _sample_points(self, n_points):
        # Use Sobol sequence sampling for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, seed=self.rng, scramble=True)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest surrogate model
        # Use a reasonable number of trees to balance accuracy and speed
        
        # Determine number of trees based on budget and dimensionality
        n_trees = min(100, max(20, int(0.5 * self.budget / (self.dim + 1))))
        
        # For small datasets, use more trees for better variance estimation
        if len(X) < 20:
            n_trees = 100
        
        # Create Random Forest with bootstrap sampling and proper randomness
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=False,  # We don't need OOB score for this use case
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Use single thread to maintain reproducibility
        )
        
        rf.fit(X, y.flatten())
        return rf

    def _acquisition_function(self, X):
        # Use an ensemble of acquisition functions weighted by their performance
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        rf = self._fit_model(self.X, self.y)
        
        # Get predictions from Random Forest
        mu = rf.predict(X)
        
        # Get uncertainty estimates (variance of predictions across trees)
        # For Random Forest, we can approximate uncertainty as variance of individual tree predictions
        if hasattr(rf, 'estimators_'):
            # Get predictions from all trees for each point
            tree_preds = np.array([tree.predict(X) for tree in rf.estimators_]).T
            sigma = np.std(tree_preds, axis=1)
        else:
            # Fallback: use a constant uncertainty estimate
            sigma = np.full(len(X), 0.1)
        
        # Expected Improvement (EI)
        best_y = np.min(self.y)
        ei_values = np.zeros_like(mu)
        
        # Handle case where sigma is zero (no uncertainty)
        mask = sigma > 0
        if np.any(mask):
            improvement = best_y - mu[mask]
            z = improvement / sigma[mask]
            ei_values[mask] = improvement * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        # Lower Confidence Bound (LCB)
        kappa = 2.0  # Exploration parameter
        lcb = mu - kappa * sigma
        
        # Thompson sampling: sample from the distribution of predictions
        # For RF, we can sample from the distribution around the mean with uncertainty
        samples = mu + self.rng.normal(0, sigma, len(X))
        thompson = -samples  # Negative because we minimize
        
        # Combine acquisition functions with weights
        acq_values = (
            self.acq_func_weights[0] * ei_values.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _select_next_points(self, batch_size):
        # Select points using a hybrid approach:
        # 1. First try to find the best point with acquisition function
        # 2. Then use local search around promising regions
        
        # Sample candidates for optimization
        n_candidates = min(1000, max(50, self.budget - self.n_evals))
        
        # Get points from current trust region if we have a center
        if self.trust_region_center is not None:
            # Generate points around the trust region center
            candidates = self._sample_points(n_candidates)
            
            # Apply trust region constraint: only consider points within radius
            distances = np.linalg.norm(candidates - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            if len(valid_indices) > 0:
                candidates = candidates[valid_indices]
            
            # If not enough candidates in trust region, expand the search
            if len(candidates) < batch_size:
                additional_candidates = self._sample_points(batch_size - len(candidates))
                candidates = np.vstack([candidates, additional_candidates])
        else:
            # First few iterations: random sampling with trust region expansion
            candidates = self._sample_points(n_candidates)
        
        # Limit candidates to reasonable size
        if len(candidates) > n_candidates:
            candidates = candidates[:n_candidates]
            
        # Evaluate acquisition function on candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points using top-k selection
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if len(X) == 0:
            return np.array([])
            
        y = []
        for point in X:
            try:
                val = func(point)
                y.append(val)
            except Exception as e:
                # Handle function evaluation errors by returning large value
                y.append(1e10)
                
        y = np.array(y).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            
        # Update trust region based on progress monitoring
        if len(self.y) > self.progress_window:
            # Check recent improvement
            recent_values = self.y[-self.progress_window:]
            old_values = self.y[:-self.progress_window]
            
            if len(old_values) > 0:
                best_old = np.min(old_values)
                best_recent = np.min(recent_values)
                
                # If no significant improvement, increment consecutive no improvement counter
                if (best_recent >= best_old - self.min_progress_improvement):
                    self.consecutive_no_improvement += 1
                else:
                    self.consecutive_no_improvement = 0
                    
                # If we've had enough non-improving steps, trigger restart
                if self.consecutive_no_improvement >= self.restart_threshold:
                    # Reset trust region to a wider scope
                    self.trust_region_radius = min(self.max_trust_region, self.trust_region_radius * 2.0)
                    self.consecutive_no_improvement = 0
                    self.restart_count += 1
                    
                    # Set center to a random point in the search space for global exploration
                    self.trust_region_center = self._sample_points(1)[0]
                    
            else:
                # If we don't have enough history, just reset if needed
                if self.consecutive_no_improvement >= self.restart_threshold:
                    self.trust_region_radius = min(self.max_trust_region, self.trust_region_radius * 2.0)
                    self.consecutive_no_improvement = 0
                    self.restart_count += 1
                    self.trust_region_center = self._sample_points(1)[0]
                
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Set trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.trust_region_center = self.X[best_idx]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, evaluate more points in parallel
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate these points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Update trust region center based on new best point
            if len(self.y) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx]
                
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
