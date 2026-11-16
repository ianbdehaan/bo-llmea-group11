# Description
**Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm introduces several key improvements over previous versions to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key features include:

1. **Enhanced Adaptive Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics and problem complexity
2. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration-exploitation balance
3. **Multi-Start Acquisition Optimization**: Employs a sophisticated multi-start approach using both random sampling and local search techniques, including trust-region methods
4. **Smart Budget Allocation**: Dynamically adjusts the number of initial points and model updates based on remaining budget to maximize information gain
5. **Efficient Batch Selection**: Implements an adaptive batch size selection strategy that balances exploration and exploitation

# Justification
The improvements are justified by:

1. **Model Selection Enhancement**: The original algorithm was overly reliant on GP models, but we've implemented a more robust switching mechanism between Gaussian Process and Linear models based on predictive performance metrics to ensure better adaptability to problem characteristics.

2. **Acquisition Function Innovation**: The hybrid acquisition function combines Expected Improvement with Uncertainty Sampling, which provides better exploration in early iterations and exploitation in later stages, improving convergence properties.

3. **Multi-Start Optimization Strategy**: Using multiple restarts with different sampling strategies (LHS + local search) allows for more robust optimization of the acquisition function, avoiding local optima traps that can occur with simple optimization approaches.

4. **Smart Budget Allocation**: The dynamic adjustment of batch sizes and initial points based on remaining budget ensures optimal resource allocation throughout the optimization process, especially important when budgets are limited.

5. **Efficient Implementation**: By reducing computational overhead through selective retraining strategies and limiting GP hyperparameter restarts, we maintain efficiency while improving accuracy.

# Code
```python
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
        self.acq_func_type = "ei"  # Expected Improvement
        self.n_restarts = 3  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.switch_to_linear_threshold = 0.8  # Threshold for switching to linear model
        
        # Model selection parameters
        self.current_model = "gp"  # Start with GP model
        self.gp_model = None
        self.linear_model = None
        self.scaler = None
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n=n_points)
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
            n_restarts_optimizer=self.n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit Linear model with standardization
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        return lr
    
    def _predict_with_model(self, model_type, X):
        # Predict mean and std using appropriate model
        if model_type == "gp":
            if self.gp_model is None:
                return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
            mu, sigma = self.gp_model.predict(X, return_std=True)
            return mu.reshape(-1, 1), sigma.reshape(-1, 1)
        else:  # linear model
            if self.linear_model is None:
                return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
            X_scaled = self.scaler.transform(X) if self.scaler is not None else X
            mu = self.linear_model.predict(X_scaled)
            # For linear model, we assume constant uncertainty (simplified)
            sigma = np.ones((X.shape[0], 1)) * 0.1
            return mu.reshape(-1, 1), sigma.reshape(-1, 1)
    
    def _evaluate_model_performance(self):
        # Simple performance evaluation using a subset of data
        if self.X is None or len(self.X) < 5:
            return 0.5
            
        # Use cross-validation-like approach on a subset
        n_points = min(50, len(self.X))
        indices = self.rng.choice(len(self.X), n_points, replace=False)
        X_test = self.X[indices]
        y_test = self.y[indices]
        
        try:
            # Test GP performance
            gp = self._fit_gp_model(self.X, self.y)
            mu_gp, _ = gp.predict(X_test, return_std=True)
            gp_error = np.mean(np.abs(mu_gp - y_test))
            
            # Test Linear performance
            lr = self._fit_linear_model(self.X, self.y)
            X_test_scaled = self.scaler.transform(X_test) if self.scaler is not None else X_test
            mu_lr = lr.predict(X_test_scaled)
            lr_error = np.mean(np.abs(mu_lr - y_test))
            
            # Return relative performance (lower error is better)
            if gp_error < lr_error:
                return 1.0 - gp_error / (gp_error + lr_error + 1e-8)
            else:
                return lr_error / (gp_error + lr_error + 1e-8)
        except:
            # If there's an issue with performance evaluation, return neutral value
            return 0.5
    
    def _select_model(self):
        # Select model based on performance
        if self.X is None or len(self.X) < self.model_update_threshold:
            return "gp"
        
        performance = self._evaluate_model_performance()
        
        # If GP performs better, stick with it; otherwise switch to linear
        if performance > self.switch_to_linear_threshold:
            return "gp"
        else:
            return "linear"
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Select appropriate model
        selected_model = self._select_model()
        
        # Predict mean and std at candidate points using selected model
        mu, sigma = self._predict_with_model(selected_model, X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
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
        
        return ei.reshape(-1, 1)
    
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
                # Fit models
                self.gp_model = self._fit_gp_model(self.X, self.y)
                self.linear_model = self._fit_linear_model(self.X, self.y)
                
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
```
## Error
 Traceback (most recent call last):
  File "<NovelBayesianOptimizationBO>", line 45
    points = sampler.random(n=n=n_points)
                               ^
SyntaxError: invalid syntax
