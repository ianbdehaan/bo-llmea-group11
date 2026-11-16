from collections.abc import Callable
from scipy.stats import qmc
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
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.progress_window = 5  # Number of recent evaluations to check for progress
        self.min_progress_improvement = 0.01  # Minimum improvement to consider progress
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling
        self.exploration_weight_schedule = []  # Dynamic exploration weight based on progress
        
        # Batch selection parameters
        self.batch_size = 1
        self.min_batch_size = 1
        self.max_batch_size = 5
        
        # Surrogate model parameters
        self.model_complexity = 'medium'  # 'low', 'medium', 'high'
        self.n_model_restarts = 2  # Number of GP hyperparameter restarts

    def _sample_points(self, n_points):
        # Use maximin Latin Hypercube sampling for better space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use a subset of data for training to keep computational cost low
        
        # Determine model complexity based on evaluations
        if len(X) < 20:
            self.model_complexity = 'low'
            n_train = min(15, len(X))
        elif len(X) < 50:
            self.model_complexity = 'medium'
            n_train = min(30, len(X))
        else:
            self.model_complexity = 'high'
            n_train = len(X)
            
        # If we have enough points, use a subset for training to keep it fast
        if len(X) > 20:
            # Select a random subset of points for training to keep it fast
            train_indices = self.rng.choice(len(X), size=n_train, replace=False)
            X_train = X[train_indices]
            y_train = y[train_indices]
        else:
            X_train = X
            y_train = y
            
        # Use ARD Matern kernel with automatic relevance detection
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # Fit GP with noise estimation and optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=self.n_model_restarts,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_train, y_train)
        return gp

    def _acquisition_function(self, X):
        # Use an ensemble of acquisition functions weighted by their performance
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
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
        # Use dynamic kappa based on progress
        if len(self.exploration_weight_schedule) > 0:
            kappa = self.exploration_weight_schedule[-1]
        else:
            kappa = 2.0  # Initial exploration weight
            
        lcb = mu - kappa * sigma
        
        # Thompson sampling: sample from posterior distribution
        samples = gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        thompson = -samples.flatten()  # Negative because we minimize
        
        # Combine acquisition functions with weights
        acq_values = (
            self.acq_func_weights[0] * ei_values.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _select_next_points(self, batch_size):
        # Select points using a hybrid approach:
        # 1. Use multi-start optimization for acquisition function
        # 2. Incorporate trust region constraints
        
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
            
        # Optimize acquisition function using multi-start approach
        best_acq_value = -np.inf
        best_point = None
        
        # Multi-start optimization with diverse starting points
        start_points = []
        
        # 1. Random starting points
        n_random_starts = min(20, max(5, batch_size * 5))
        random_starts = self._sample_points(n_random_starts)
        start_points.extend(random_starts)
        
        # 2. Trust region center if exists
        if self.trust_region_center is not None:
            start_points.append(self.trust_region_center)
            
        # 3. Best point so far
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            start_points.append(self.X[best_idx])
        
        # Ensure we have enough starting points
        while len(start_points) < n_random_starts:
            start_points.append(self._sample_points(1)[0])
            
        start_points = np.array(start_points[:n_random_starts])
        
        # Optimize acquisition function from each starting point using L-BFGS-B
        for start_point in start_points:
            try:
                # Define acquisition function for optimization
                def acq_func(x):
                    x = x.reshape(1, -1)
                    acq_val = self._acquisition_function(x)
                    return -float(acq_val)  # Negative because we minimize
                
                # Optimize using L-BFGS-B
                res = minimize(
                    acq_func, 
                    start_point, 
                    method='L-BFGS-B', 
                    bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                    options={'maxiter': 20}
                )
                
                if res.success:
                    candidate_x = res.x.reshape(1, -1)
                    candidate_acq = self._acquisition_function(candidate_x)[0]
                    
                    if candidate_acq > best_acq_value:
                        best_acq_value = candidate_acq
                        best_point = candidate_x[0]
            except Exception:
                continue
                
        # If we didn't find a good point, use random selection
        if best_point is None:
            # Select the best among candidates based on acquisition function values
            acq_values = self._acquisition_function(candidates)
            top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
            selected_points = candidates[top_indices]
        else:
            # Use the optimized point plus additional random points for diversity
            selected_points = [best_point]
            if batch_size > 1:
                # Add diverse random points from trust region
                remaining_points = self._sample_points(batch_size - 1)
                selected_points.extend(remaining_points[:batch_size-1])
            
            selected_points = np.array(selected_points)
        
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
            
        # Update exploration weight based on progress monitoring
        if len(self.y) > self.progress_window:
            # Check recent improvement
            recent_values = self.y[-self.progress_window:]
            old_values = self.y[:-self.progress_window]
            
            if len(old_values) > 0:
                best_old = np.min(old_values)
                best_recent = np.min(recent_values)
                
                # Calculate progress ratio
                if best_old > 0:
                    progress_ratio = (best_old - best_recent) / best_old
                else:
                    progress_ratio = (best_old - best_recent) / (best_old + 1e-8)
                
                # Adjust exploration weight based on progress
                if progress_ratio < 0.001:  # Very little progress
                    new_weight = 3.0  # Increase exploration
                elif progress_ratio < 0.01:  # Slow progress
                    new_weight = 2.5  # Moderate exploration
                elif progress_ratio < 0.05:  # Moderate progress
                    new_weight = 2.0  # Balanced exploration
                else:  # Good progress
                    new_weight = 1.0  # Reduce exploration
                
                self.exploration_weight_schedule.append(new_weight)
            else:
                self.exploration_weight_schedule.append(2.0)  # Default weight
        else:
            self.exploration_weight_schedule.append(2.0)  # Default weight
            
    def _adjust_batch_size(self):
        # Dynamically adjust batch size based on remaining budget and problem characteristics
        remaining_budget = self.budget - self.n_evals
        
        if remaining_budget <= 10:
            self.batch_size = 1
        elif remaining_budget <= 30:
            self.batch_size = min(2, self.max_batch_size)
        elif remaining_budget <= 60:
            self.batch_size = min(3, self.max_batch_size)
        else:
            self.batch_size = min(5, self.max_batch_size)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Set trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.trust_region_center = self.X[best_idx]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Adjust batch size based on remaining budget
            self._adjust_batch_size()
            
            # Limit batch size to remaining budget
            batch_size = min(self.batch_size, self.budget - self.n_evals)
            
            if batch_size <= 0:
                break
                
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
