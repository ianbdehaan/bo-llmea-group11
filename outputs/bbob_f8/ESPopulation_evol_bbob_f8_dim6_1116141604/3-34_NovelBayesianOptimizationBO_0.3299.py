from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import Ridge
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
        self.n_restarts = 5  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_type = "gp"  # Current model type: "gp" or "linear"
        self.model_switch_threshold = 0.8  # Threshold to switch from GP to linear model
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
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
            n_restarts_optimizer=min(5, self.n_restarts),  # Reduced for efficiency
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit a simple linear model with Ridge regression for faster computation
        # This is used as a fallback when GP performance degrades
        ridge = Ridge(alpha=1.0, random_state=int(self.rng.integers(2**31 - 1)))
        ridge.fit(X, y)
        return ridge
    
    def _get_model_predictions(self, X):
        # Get predictions from the current model
        if self.model_type == "gp":
            # For GP, we need to fit a new model each time for accurate predictions
            gp = self._fit_gp_model(self.X, self.y)
            mu, sigma = gp.predict(X, return_std=True)
            return mu, sigma
        else:
            # For linear model
            lr = self._fit_linear_model(self.X, self.y)
            mu = lr.predict(X)
            # Simple approximation for uncertainty (constant variance)
            sigma = np.full(mu.shape, 0.1)
            return mu, sigma
    
    def _model_performance_check(self):
        # Check if we should switch model types based on recent performance
        if self.X is None or self.X.shape[0] < 5:
            return False
            
        # Sample points to evaluate current model performance
        n_samples = min(100, self.X.shape[0])
        sample_indices = self.rng.choice(self.X.shape[0], n_samples, replace=False)
        X_sample = self.X[sample_indices]
        y_sample = self.y[sample_indices]
        
        # Evaluate GP performance
        gp = self._fit_gp_model(X_sample, y_sample)
        y_pred_gp = gp.predict(X_sample)
        gp_error = mean_squared_error(y_sample, y_pred_gp)
        
        # Evaluate linear model performance
        lr = self._fit_linear_model(X_sample, y_sample)
        y_pred_lr = lr.predict(X_sample)
        lr_error = mean_squared_error(y_sample, y_pred_lr)
        
        # If linear model performs better, switch to it
        if lr_error < gp_error * self.model_switch_threshold:
            return True
        return False
    
    def _acquisition_function(self, X):
        # Expected Improvement with uncertainty sampling component
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Get predictions from current model
        mu, sigma = self._get_model_predictions(X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add uncertainty sampling component to encourage exploration
        uncertainty_factor = 0.3  # Blend factor between EI and uncertainty
        uncertainty = sigma  # Higher uncertainty = more exploration
        
        # Combine acquisition components
        acq_value = ei + uncertainty_factor * uncertainty
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        exploitation_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.8)
        acq_value *= exploitation_factor
        
        return acq_value.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, n_candidates):
        # Use hybrid approach to optimize acquisition function
        # Start with multiple random initial points for better exploration
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points
    
    def _select_next_points(self, batch_size):
        # Use hybrid strategy to optimize acquisition function
        n_candidates = max(50, batch_size * 20)
        
        # Optimize acquisition function
        next_points = self._optimize_acquisition_function(n_candidates)
        
        return next_points
    
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
        
        # Adaptive batch size adjustment
        if self.budget > 200:
            self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
        else:
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Check if we should switch models based on performance
                if self.n_evals > 50 and self._model_performance_check():
                    self.model_type = "linear"
                
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Update batch size based on remaining budget
                if self.budget > 200:
                    self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
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
