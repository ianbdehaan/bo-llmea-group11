from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EfficientBayesianOptimizationBO:
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
        self.n_restarts = 2  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_type = "gp"  # Default model type
        self.model_switch_threshold = 0.5  # Threshold to switch between models
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit Gaussian Process with Matern kernel or Linear model based on performance
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
                n_restarts_optimizer=self.n_restarts,
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            return gp
        else:
            # Fit Linear model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            return (lr, scaler)
    
    def _predict_model(self, model, X):
        # Predict using either GP or Linear model
        if isinstance(model, GaussianProcessRegressor):
            return model.predict(X, return_std=False)
        else:
            lr, scaler = model
            X_scaled = scaler.transform(X)
            return lr.predict(X_scaled)
    
    def _evaluate_model_performance(self, model, X, y):
        # Evaluate model performance using cross-validation
        if self.model_type == "gp":
            # For GP, use the log marginal likelihood as a proxy for performance
            try:
                return -model.log_marginal_likelihood_value_
            except:
                return float('inf')
        else:
            # For linear model, use cross-validation score
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LinearRegression()
            scores = cross_val_score(lr, X_scaled, y, cv=min(5, X.shape[0]), scoring='neg_mean_squared_error')
            return -np.mean(scores)  # Return negative MSE for minimization
    
    def _switch_model_if_needed(self, X, y):
        # Switch between GP and Linear model based on performance
        if X.shape[0] < 15:
            return self.model_type
            
        try:
            # Try both models
            gp_model = self._fit_model(X, y)
            linear_model = self._fit_model(X, y)  # This will be a different approach
            
            # Evaluate performance of both models
            gp_performance = self._evaluate_model_performance(gp_model, X, y)
            linear_performance = self._evaluate_model_performance(linear_model, X, y)
            
            # Switch to the better performing model (lower is better)
            if gp_performance < linear_performance:
                return "gp"
            else:
                return "linear"
        except:
            # If there's any error, stick with current model
            return self.model_type
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        gp = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more gradually
        if self.n_evals < self.budget * 0.3:
            exploitation_factor = 1.0  # More exploration in early phase
        elif self.n_evals < self.budget * 0.7:
            exploitation_factor = 0.5  # Balanced
        else:
            exploitation_factor = 0.1  # More exploitation in late phase
            
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(100, batch_size * 50)
        
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
        
        # Adaptive batch size adjustment - more aggressive in early stages
        if self.budget > 100:
            self.batch_size = min(5, max(1, int(np.ceil(self.budget / 50))))
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
                
                # Update batch size based on remaining budget and progress
                if self.budget > 100:
                    # Gradually reduce batch size to be more conservative in later iterations
                    progress = self.n_evals / self.budget
                    if progress < 0.3:
                        self.batch_size = min(5, max(1, int(np.ceil(self.budget / 50))))
                    elif progress < 0.7:
                        self.batch_size = min(3, max(1, int(np.ceil(self.budget / 100))))
                    else:
                        self.batch_size = 1
                    
                # Switch model if needed (but not too frequently)
                if self.X.shape[0] > 20 and self.n_evals % 10 == 0:
                    self.model_type = self._switch_model_if_needed(self.X, self.y)
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
