from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
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
        self.switch_threshold = 0.2  # Threshold to switch between models
        self.adaptive_factor = 1.0  # Adaptive factor for exploration-exploitation balance
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on performance
        if self.model_type == "gp":
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
                n_restarts_optimizer=min(5, max(1, int(np.ceil(self.n_restarts * 0.5)))),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            return gp
        else:
            # Linear model (Ridge regression)
            ridge = Ridge(alpha=1.0, random_state=int(self.rng.integers(2**31 - 1)))
            ridge.fit(X, y)
            return ridge
    
    def _predict_model(self, model, X):
        # Predict mean and std for GP or return prediction for linear model
        if hasattr(model, 'predict'):
            if self.model_type == "gp":
                return model.predict(X, return_std=True)
            else:
                return model.predict(X), None
        else:
            return model.predict(X), None
    
    def _evaluate_model_performance(self, X_train, y_train, X_val, y_val):
        # Evaluate performance of both models on validation set
        if self.model_type == "gp":
            # For GP model, we'll use a simple approach: compute MSE on validation data
            gp = self._fit_model(X_train, y_train)
            mu, _ = self._predict_model(gp, X_val)
            mse_gp = mean_squared_error(y_val, mu)
            return mse_gp
        else:
            # For linear model, we'll use a simple approach: compute MSE on validation data  
            linear_model = self._fit_model(X_train, y_train)
            pred = self._predict_model(linear_model, X_val)[0]
            mse_linear = mean_squared_error(y_val, pred)
            return mse_linear
    
    def _switch_model(self, X_train, y_train, X_val, y_val):
        # Switch to linear model if it performs better than GP
        if self.X.shape[0] > 50:  # Only switch after sufficient data points
            try:
                gp_mse = self._evaluate_model_performance(X_train, y_train, X_val, y_val)
                # Fit a linear model for comparison
                linear_model = Ridge(alpha=1.0, random_state=int(self.rng.integers(2**31 - 1)))
                linear_model.fit(X_train, y_train)
                pred = linear_model.predict(X_val)
                mse_linear = mean_squared_error(y_val, pred)
                
                # Switch to linear model if it's significantly better
                if mse_linear < gp_mse * (1.0 - self.switch_threshold):
                    self.model_type = "linear"
                elif gp_mse < mse_linear * (1.0 - self.switch_threshold):
                    self.model_type = "gp"
            except:
                # If we encounter issues, keep current model
                pass
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function combining Expected Improvement with Uncertainty Sampling
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        model = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict_model(model, X)
        
        # Avoid numerical issues with zero std
        if sigma is not None:
            sigma = np.maximum(sigma, 1e-8)
        else:
            # For linear models, we approximate uncertainty
            sigma = np.ones((X.shape[0], 1)) * 1e-4
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add uncertainty sampling component (variance-based exploration)
        uncertainty_component = sigma * 0.5  # Scale factor
        
        # Combine both components
        combined_acq = ei + uncertainty_component
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        self.adaptive_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.8)
        combined_acq *= self.adaptive_factor
        
        return combined_acq.reshape(-1, 1)
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(50, batch_size * 20)
        
        # Sample candidates using LHS
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points
    
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
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Occasionally switch models based on performance
                if self.X.shape[0] > 30 and self.n_evals % 10 == 0:
                    # Use a validation split to determine model quality
                    n_val = max(5, min(20, int(np.ceil(self.X.shape[0] * 0.2))))
                    val_indices = self.rng.choice(self.X.shape[0], n_val, replace=False)
                    X_val = self.X[val_indices]
                    y_val = self.y[val_indices]
                    # Remove validation points from training set
                    train_mask = np.ones(self.X.shape[0], dtype=bool)
                    train_mask[val_indices] = False
                    X_train = self.X[train_mask]
                    y_train = self.y[train_mask]
                    self._switch_model(X_train, y_train, X_val, y_val)
                
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
