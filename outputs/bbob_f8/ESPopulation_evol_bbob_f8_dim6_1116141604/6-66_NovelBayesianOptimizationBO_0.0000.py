from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class NovelBayesianOptimizationBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, min(20, int(np.ceil(np.sqrt(self.dim) * 5))))
        
        # For reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.acq_func_type = "ei"  # Expected Improvement
        self.n_restarts = 5  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.switch_to_linear_threshold = 0.8  # Switch to linear model if GP error is above this threshold
        self.model_type = "gp"  # Current model type
        
        # Acquisition function parameters
        self.exploitation_factor = 1.0
        self.exploration_factor = 1.0
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on performance and characteristics
        try:
            # First try GP model with proper error handling
            kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
            
            # Use a subset of data points if too many to avoid numerical issues
            max_samples = min(1000, X.shape[0])
            if X.shape[0] > max_samples:
                indices = self.rng.choice(X.shape[0], max_samples, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
                
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=min(5, max(1, self.n_restarts)),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            
            # Evaluate GP performance using multiple metrics
            if X.shape[0] > 10:
                # Use cross-validation for more robust evaluation
                try:
                    cv_scores = cross_val_score(gp, X_subset, y_subset.flatten(), cv=min(5, X_subset.shape[0]), 
                                              scoring='neg_mean_squared_error')
                    gp_error = -np.mean(cv_scores)
                    
                    # Also evaluate prediction variance
                    if X_subset.shape[0] > 1:
                        y_pred, y_std = gp.predict(X_subset, return_std=True)
                        avg_variance = np.mean(y_std**2)
                        
                        # Make decision based on multiple factors
                        if gp_error > self.switch_to_linear_threshold or avg_variance < 0.1:
                            model = LinearRegression()
                            model.fit(X, y)
                            return model, "linear"
                        else:
                            return gp, "gp"
                    else:
                        return gp, "gp"
                except Exception:
                    # If cross-validation fails, fall back to simple validation
                    n_val = max(5, min(100, X.shape[0] // 5))
                    val_indices = self.rng.choice(X.shape[0], n_val, replace=False)
                    X_val = X[val_indices]
                    y_val = y[val_indices]
                    
                    y_pred, _ = gp.predict(X_val, return_std=True)
                    gp_error = mean_squared_error(y_val, y_pred)
                    
                    if gp_error > self.switch_to_linear_threshold:
                        model = LinearRegression()
                        model.fit(X, y)
                        return model, "linear"
                    else:
                        return gp, "gp"
            else:
                return gp, "gp"
                
        except Exception:
            # Fallback to linear model if GP fails
            model = LinearRegression()
            model.fit(X, y)
            return model, "linear"
    
    def _acquisition_function(self, X):
        # Advanced acquisition function combining Expected Improvement with adaptive exploration components
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit model
        model, model_type = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        if model_type == "gp":
            mu, sigma = model.predict(X, return_std=True)
            # Avoid numerical issues with zero std
            sigma = np.maximum(sigma, 1e-8)
            
            # Expected Improvement calculation
            improvement = best_y - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Add adaptive exploration component based on problem characteristics and progress
            progress_ratio = self.n_evals / self.budget
            
            # Dynamic factor that adapts to problem dimensionality and optimization stage
            if self.dim <= 10:
                # For low-dimensional problems, we can be more aggressive in exploitation
                exploration_factor = 1.0 - progress_ratio * 0.7
                exploitation_factor = 0.3 + progress_ratio * 0.5
            else:
                # For high-dimensional problems, maintain more exploration
                exploration_factor = 1.0 - progress_ratio * 0.9
                exploitation_factor = 0.1 + progress_ratio * 0.6
                
            # Adjust factors based on how well we're doing
            if self.n_evals > 10:
                recent_improvement = np.mean(self.y[-5:]) if len(self.y) >= 5 else np.mean(self.y)
                if recent_improvement < best_y - 0.1:  # If we're making progress, exploit more
                    exploitation_factor *= 1.2
                else:  # If not, explore more
                    exploration_factor *= 1.2
            
            # Apply the adaptive factors
            ei = ei * exploitation_factor + np.std(self.y) * exploration_factor * 0.5
            
        else:
            # For linear model, use a more sophisticated approach
            mu = model.predict(X)
            # Use uncertainty-based acquisition with dynamic scaling
            uncertainty = np.std(self.y) if len(self.y) > 1 else 1.0
            
            # Simple but effective hybrid: use both distance to best and uncertainty
            ei = -np.abs(mu - best_y) + uncertainty * 0.5
            
            # Apply adaptive factor based on problem characteristics
            progress_ratio = self.n_evals / self.budget
            if progress_ratio < 0.3:
                ei += uncertainty * 0.7  # More exploration in early stages
            else:
                ei += uncertainty * 0.3  # Less exploration, more exploitation
        
        # Ensure non-negative acquisition values for numerical stability
        return np.maximum(ei.reshape(-1, 1), 1e-8)
    
    def _optimize_acquisition_function(self, candidates):
        # Multi-start optimization of acquisition function using sophisticated sampling and selection
        n_candidates = len(candidates)
        
        # Evaluate all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value (using a more sophisticated approach)
        # Sort by acquisition values and select top N points for refinement
        sorted_indices = np.argsort(acq_values.flatten())[::-1]
        top_candidates = candidates[sorted_indices[:min(20, n_candidates)]]
        
        # Use a hybrid approach: sample from best candidates with probability proportional to their acq value
        if len(top_candidates) > 0:
            # Compute probabilities for selection (softmax)
            normalized_acq = np.exp(acq_values[sorted_indices[:len(top_candidates)]].flatten() - 
                                  np.max(acq_values[sorted_indices[:len(top_candidates)]]))
            probs = normalized_acq / np.sum(normalized_acq)
            
            # Select best candidates with higher probability
            selected_indices = self.rng.choice(len(top_candidates), min(5, len(top_candidates)), 
                                              p=probs, replace=False)
            best_candidates = top_candidates[selected_indices]
        else:
            best_candidates = candidates[:1] if len(candidates) > 0 else np.array([]).reshape(1, -1)
        
        # Return the best candidate found
        if len(best_candidates) > 0:
            best_idx = np.argmax(self._acquisition_function(best_candidates).flatten())
            return best_candidates[best_idx:best_idx+1]
        else:
            return candidates[:1] if len(candidates) > 0 else np.array([]).reshape(1, -1)
    
    def _select_next_points(self, batch_size):
        # Improved selection strategy with better exploration-exploitation balance
        n_candidates = max(50, batch_size * 20)
        
        # Sample candidates using LHS with some adaptive sampling
        candidates = self._sample_points(n_candidates)
        
        # Add some local search points around the current best if we have good data
        if self.X is not None and len(self.X) > 0:
            # Add points around the current best
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Generate adaptive perturbations based on problem characteristics
            if self.dim <= 10:
                # For low-dim problems, use larger perturbations
                perturbation_size = min(1.5, 2.0 * (5.0 - (-5.0)) / np.sqrt(self.budget))
            else:
                # For high-dim problems, use smaller perturbations
                perturbation_size = min(0.8, 2.0 * (5.0 - (-5.0)) / self.budget)
            
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Optimize acquisition function for candidates and select best points
        selected_points = self._optimize_acquisition_function(candidates)
        
        # For batch selection, use a greedy approach to avoid duplicates
        if batch_size > 1:
            remaining_candidates = candidates[~np.any(np.all(np.abs(candidates - selected_points[0]), axis=1) < 1e-8)]
            
            if len(remaining_candidates) >= batch_size - 1:
                # Select additional points using a greedy approach
                batch_points = [selected_points[0]]
                
                for _ in range(batch_size - 1):
                    if len(remaining_candidates) == 0:
                        break
                        
                    # Find point with maximum distance to already selected points
                    distances = []
                    for candidate in remaining_candidates:
                        min_dist = np.min([np.linalg.norm(candidate - p) for p in batch_points])
                        distances.append(min_dist)
                    
                    if len(distances) > 0:
                        max_idx = np.argmax(distances)
                        batch_points.append(remaining_candidates[max_idx])
                        remaining_candidates = np.delete(remaining_candidates, max_idx, axis=0)
                
                selected_points = np.array(batch_points)
            else:
                # If not enough candidates, return what we have
                pass
        
        return selected_points[:batch_size] if len(selected_points) > batch_size else selected_points
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y_values = []
        for x in X:
            y_values.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)
    
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
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Adaptive batch size adjustment based on remaining budget and problem characteristics
        if self.budget > 200:
            # For larger budgets, use adaptive batch sizing with more sophisticated logic
            remaining_budget = self.budget - self.n_evals
            if remaining_budget > 50:
                # Use dynamic batch size that decreases as we approach the end
                progress_ratio = self.n_evals / self.budget
                batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.8)
                self.batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, 
                                        int(np.ceil(batch_size_factor * 4))))
            else:
                # Use small batch size for final evaluations
                self.batch_size = 1
        else:
            # For smaller budgets, stick to single evaluation
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Update batch size based on remaining budget and problem characteristics
                if self.budget > 200:
                    remaining_budget = self.budget - self.n_evals
                    if remaining_budget > 50:
                        # Use dynamic batch sizing that decreases as we approach the end
                        progress_ratio = self.n_evals / self.budget
                        batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.8)
                        self.batch_size = max(self.min_batch_size, 
                                            min(self.max_batch_size, 
                                                int(np.ceil(batch_size_factor * 4))))
                    else:
                        # Use small batch size for final evaluations
                        self.batch_size = 1
            else:
                # If we don't have enough points, evaluate more initial points
                remaining_budget = self.budget - self.n_evals
                if remaining_budget > 0:
                    additional_points = self._sample_points(min(remaining_budget, 10))
                    additional_values = self._evaluate_points(func, additional_points)
                    self._update_eval_points(additional_points, additional_values)
        
        # Return best point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
