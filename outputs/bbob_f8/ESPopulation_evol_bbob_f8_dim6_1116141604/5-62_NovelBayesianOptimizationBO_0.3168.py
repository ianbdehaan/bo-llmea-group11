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
        self.acq_func_type = "hybrid"  # Hybrid acquisition function combining EI and uncertainty sampling
        self.n_restarts = 10  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_type = "gp"  # Start with Gaussian Process model
        self.model_switch_threshold = 0.8  # Threshold to switch between models based on performance
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on problem characteristics
        if self.model_type == "gp":
            # Fit Gaussian Process with Matern kernel
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
                n_restarts_optimizer=min(5, self.n_restarts),  # Reduce restarts for efficiency
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            return gp
        else:
            # Fit Linear model with standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            return (lr, scaler)
    
    def _predict_model(self, model, X):
        # Predict using the appropriate model type
        if isinstance(model, GaussianProcessRegressor):
            return model.predict(X, return_std=False)
        else:
            # Linear model
            lr, scaler = model
            X_scaled = scaler.transform(X)
            return lr.predict(X_scaled)
    
    def _get_model_performance(self, model, X_test, y_test):
        # Calculate predictive performance (negative MSE for comparison)
        if isinstance(model, GaussianProcessRegressor):
            predictions = model.predict(X_test, return_std=False)
        else:
            predictions = self._predict_model(model, X_test)
        
        mse = np.mean((predictions.flatten() - y_test.flatten()) ** 2)
        return -mse  # Return negative MSE for comparison (higher is better)
    
    def _switch_model(self, X, y):
        # Switch to linear model if GP performs poorly
        if self.X.shape[0] < 10:
            return "gp"  # Use GP for small datasets
            
        # Fit both models and compare performance on a validation set
        n_val = min(20, max(5, X.shape[0] // 5))
        val_indices = self.rng.choice(X.shape[0], n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(X.shape[0]), val_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Fit GP model
        gp_model = self._fit_model(X_train, y_train)
        gp_performance = self._get_model_performance(gp_model, X_val, y_val)
        
        # Fit Linear model  
        lin_model = self._fit_model(X_train, y_train)
        lin_performance = self._get_model_performance(lin_model, X_val, y_val)
        
        # Switch if linear performs significantly better
        if lin_performance > gp_performance * self.model_switch_threshold:
            return "linear"
        else:
            return "gp"
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function combining Expected Improvement with Uncertainty Sampling
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        gp_model = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = gp_model.predict(X, return_std=True)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Uncertainty sampling component (add uncertainty to acquisition)
        uncertainty_component = sigma
        
        # Hybrid combination with adaptive weight based on iteration progress
        progress_ratio = self.n_evals / self.budget
        ei_weight = max(0.3, 1.0 - progress_ratio * 0.7)  # Start with more exploration
        uncertainty_weight = 1.0 - ei_weight
        
        hybrid_acq = ei_weight * ei + uncertainty_weight * uncertainty_component
        
        return hybrid_acq.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, n_candidates):
        # Multi-start optimization of acquisition function with hybrid strategy
        # Start with LHS sampling
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
        
        # Optimize acquisition function using multi-start approach
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
        
        # Adaptive batch size adjustment and model switching
        if self.budget > 200:
            self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
        else:
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Check if we should switch models based on performance
                if self.X.shape[0] > 20:
                    self.model_type = self._switch_model(self.X, self.y)
                
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
