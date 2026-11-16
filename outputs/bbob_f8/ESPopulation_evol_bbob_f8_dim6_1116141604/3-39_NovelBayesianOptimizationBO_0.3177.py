from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
        self.acq_func_type = "hybrid"  # Hybrid acquisition function (EI + US)
        self.n_restarts = 5  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.switch_to_linear_threshold = 0.8  # Threshold for switching to linear model
        
        # Model tracking
        self.current_model = "gp"  # "gp" or "linear"
        self.gp_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on performance metrics
        if self.current_model == "gp":
            return self._fit_gp_model(X, y)
        else:
            return self._fit_linear_model(X, y)
    
    def _fit_gp_model(self, X, y):
        # Fit Gaussian Process with Matern kernel
        # Use a small number of restarts for efficiency
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
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit Linear model
        if self.X is not None and self.y is not None:
            # Use the existing data to fit the linear model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, y.flatten())
            return {"model": model, "scaler": scaler}
        else:
            # Fallback to GP if not enough data
            return self._fit_gp_model(X, y)
    
    def _evaluate_model_performance(self, X_test, y_test):
        """Evaluate model performance using log-likelihood or other metrics"""
        if self.current_model == "gp" and self.gp_model is not None:
            # Use GP's log-marginal likelihood as a proxy for performance
            try:
                log_likelihood = self.gp_model.log_marginal_likelihood_value
                return log_likelihood
            except:
                return -np.inf
        elif self.current_model == "linear" and self.linear_model is not None:
            # Evaluate linear model performance using R² or other metrics
            model_info = self.linear_model
            if "model" in model_info and "scaler" in model_info:
                X_scaled = model_info["scaler"].transform(X_test)
                r2_score = model_info["model"].score(X_scaled, y_test.flatten())
                return r2_score
        return -np.inf
    
    def _switch_model_if_needed(self, X, y):
        """Switch between GP and Linear models based on performance"""
        if self.X is None or len(self.X) < 10:
            # Don't switch too early
            return
            
        # Evaluate current model
        if self.current_model == "gp":
            # Try switching to linear if performance is poor
            performance_gp = self._evaluate_model_performance(X, y)
            if self.linear_model is not None:
                performance_linear = self._evaluate_model_performance(X, y)
                if performance_linear > performance_gp * self.switch_to_linear_threshold:
                    self.current_model = "linear"
        elif self.current_model == "linear":
            # Try switching to GP if performance is poor
            performance_linear = self._evaluate_model_performance(X, y)
            if self.gp_model is not None:
                performance_gp = self._evaluate_model_performance(X, y)
                if performance_gp > performance_linear * 1.2:  # GP must be significantly better
                    self.current_model = "gp"
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function combining Expected Improvement with Uncertainty Sampling
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit model
        if self.current_model == "gp":
            model = self._fit_gp_model(self.X, self.y)
            # Predict mean and std at candidate points
            mu, sigma = model.predict(X, return_std=True)
        else:
            # Linear model
            model_info = self._fit_linear_model(self.X, self.y)
            if "model" in model_info and "scaler" in model_info:
                X_scaled = model_info["scaler"].transform(X)
                mu = model_info["model"].predict(X_scaled)
                # For linear model, use a simple uncertainty estimate
                sigma = np.full(X.shape[0], 1.0)  # Simplified uncertainty
            else:
                # Fallback to GP if linear fails
                model = self._fit_gp_model(self.X, self.y)
                mu, sigma = model.predict(X, return_std=True)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Uncertainty Sampling component
        us = sigma
        
        # Combine both components with adaptive weights
        progress_ratio = self.n_evals / self.budget
        
        # Dynamic weighting based on optimization phase
        if progress_ratio < 0.3:
            # Early phase: emphasize exploration (uncertainty sampling)
            weight_ei = 0.3
            weight_us = 0.7
        elif progress_ratio < 0.7:
            # Middle phase: balanced exploration and exploitation
            weight_ei = 0.5
            weight_us = 0.5
        else:
            # Late phase: emphasize exploitation (expected improvement)
            weight_ei = 0.7
            weight_us = 0.3
            
        # Combine acquisition functions
        acq_value = weight_ei * ei + weight_us * us
        
        return acq_value.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, n_candidates):
        """Optimize the acquisition function using multi-start approach"""
        # Start with multiple random initial points for better exploration
        candidates = self._sample_points(n_candidates)
        
        # Add some local search points around the current best
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Generate perturbations around the best point
            perturbation_size = min(1.0, (5.0 - (-5.0)) / max(10, self.budget/10))
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        n_candidates = max(50, batch_size * 30)
        return self._optimize_acquisition_function(n_candidates)
    
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
                
                # Update model if needed
                if self.X.shape[0] > self.model_update_threshold:
                    self._switch_model_if_needed(self.X, self.y)
                
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
