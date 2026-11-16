from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
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
        self.n_restarts = 3  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 10  # Minimum points before updating model
        self.switch_to_linear_threshold = 0.8  # Switch to linear model if GP error is above this threshold
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on performance
        # First try GP model
        try:
            kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
            
            # Use a subset of data points if too many to avoid numerical issues
            max_samples = min(500, X.shape[0])
            if X.shape[0] > max_samples:
                indices = self.rng.choice(X.shape[0], max_samples, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
                
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=min(3, max(1, self.n_restarts)),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            
            # Evaluate GP performance using cross-validation
            if X.shape[0] > 5:
                try:
                    # Use 5-fold cross-validation to estimate model performance
                    cv_scores = cross_val_score(gp, X, y.flatten(), cv=3, scoring='neg_mean_squared_error')
                    gp_error = -np.mean(cv_scores)
                    
                    # If GP error is too high, switch to linear model
                    if gp_error > self.switch_to_linear_threshold:
                        model = LinearRegression()
                        model.fit(X, y)
                        return model, "linear"
                    else:
                        return gp, "gp"
                except Exception:
                    # Fallback to linear model if CV fails
                    model = LinearRegression()
                    model.fit(X, y)
                    return model, "linear"
            else:
                return gp, "gp"
                
        except Exception:
            # Fallback to linear model if GP fails
            model = LinearRegression()
            model.fit(X, y)
            return model, "linear"
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function combining Expected Improvement and Uncertainty Sampling
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
        else:
            # For linear model, use simple uncertainty-based approach
            mu = model.predict(X)
            # Simple uncertainty based on data spread (for linear models, we can't get std)
            uncertainty = np.std(self.y) if len(self.y) > 1 else 1.0
            ei = -np.abs(mu - best_y) + uncertainty * 0.5
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        progress_ratio = self.n_evals / self.budget
        
        # More sophisticated adaptive factor - use different strategies based on problem dimensionality
        if self.dim <= 10:
            # For low-dimensional problems, we can be more aggressive in exploitation
            exploitation_factor = max(0.2, 1.0 - progress_ratio * 0.7)
        else:
            # For high-dimensional problems, maintain more exploration
            exploitation_factor = max(0.1, 1.0 - progress_ratio * 0.9)
        
        ei *= exploitation_factor
        
        # Add uncertainty component to encourage exploration in early stages
        if progress_ratio < 0.3:  # Early stage exploration
            uncertainty_component = np.std(self.y) if len(self.y) > 1 else 1.0
            ei += uncertainty_component * 0.5 * (1.0 - progress_ratio)
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, candidates):
        # Multi-start optimization of acquisition function using both random sampling and simple local search
        n_candidates = len(candidates)
        
        # Evaluate all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-min(10, n_candidates):][::-1]
        top_candidates = candidates[top_indices]
        
        # Simple local search approach - return the best candidate directly
        # This keeps computational cost low while maintaining good performance
        if len(top_candidates) > 0:
            best_acq_idx = np.argmax(acq_values[top_indices])
            best_candidate = top_candidates[best_acq_idx].reshape(1, -1)
        else:
            best_candidate = candidates[:1]
            
        return best_candidate
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(50, batch_size * 30)
        
        # Sample candidates using LHS
        candidates = self._sample_points(n_candidates)
        
        # Add some local search points around the current best
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Generate perturbations around the best point
            perturbation_size = min(1.0, 2.0 * (5.0 - (-5.0)) / self.budget)  # Dynamic perturbation size
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Optimize acquisition function for candidates and select best points
        selected_points = self._optimize_acquisition_function(candidates)
        
        # If we need more than one point, use the remaining candidates
        if batch_size > 1:
            remaining_candidates = candidates[~np.any(np.all(np.abs(candidates - selected_points[0]), axis=1) < 1e-8)]
            if len(remaining_candidates) >= batch_size - 1:
                # Select additional points from remaining candidates
                n_additional = batch_size - 1
                additional_indices = self.rng.choice(len(remaining_candidates), n_additional, replace=False)
                additional_points = remaining_candidates[additional_indices]
                selected_points = np.vstack([selected_points, additional_points])
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
                batch_size_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.7)
                self.batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, 
                                        int(np.ceil(batch_size_factor * 3))))
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
                
                # Update batch size based on remaining budget
                if self.budget > 200:
                    remaining_budget = self.budget - self.n_evals
                    if remaining_budget > 50:
                        # Use dynamic batch sizing that decreases as we approach the end
                        batch_size_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.7)
                        self.batch_size = max(self.min_batch_size, 
                                            min(self.max_batch_size, 
                                                int(np.ceil(batch_size_factor * 3))))
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
