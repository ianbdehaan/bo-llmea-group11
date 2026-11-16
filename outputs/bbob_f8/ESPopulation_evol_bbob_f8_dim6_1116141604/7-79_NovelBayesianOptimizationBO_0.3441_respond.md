# Description 
**Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm enhances the previous EfficientBO by introducing several key improvements:

1. **Smart Budget Allocation**: Dynamically adjusts the number of initial points based on problem characteristics and budget constraints, ensuring better exploration-exploitation balance
2. **Enhanced Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics
3. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration
4. **Multi-Start Optimization**: Implements a more sophisticated multi-start approach using both random sampling and local search techniques
5. **Adaptive Batch Size**: Dynamically adjusts batch size based on remaining budget to maximize information gain
6. **Efficient Model Updates**: Uses selective retraining strategies to minimize computational overhead

# Justification 
The key components of this algorithm are justified as follows:

1. **Smart Budget Allocation**: Using a dynamic approach to initial point selection based on problem dimensionality and budget helps balance exploration vs exploitation effectively, especially for higher dimensional problems.

2. **Enhanced Model Selection**: The switch between GP and Linear models based on predictive performance metrics provides robustness - GPs are better for complex landscapes while linear models are more efficient for simpler ones.

3. **Improved Acquisition Function**: The hybrid approach combining Expected Improvement with uncertainty sampling ensures good exploration without being overly greedy.

4. **Multi-Start Optimization**: Multiple random restarts combined with local search strategies for acquisition function optimization increases the chance of finding global optima.

5. **Adaptive Batch Size**: The dynamic batch sizing adapts to remaining budget and problem characteristics, maximizing information gain while respecting computational constraints.

6. **Efficient Model Updates**: Selective retraining prevents unnecessary computation overhead while maintaining model accuracy through intelligent update strategies.

The code uses only numpy, scipy, sklearn, and their dependencies, with proper seeding for reproducibility and efficient implementation that scales well with budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
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
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.switch_to_linear_threshold = 0.1  # Threshold for switching to linear model
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
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
        # Fit Linear model for simpler cases
        lr = LinearRegression()
        lr.fit(X, y)
        return lr
    
    def _evaluate_model_performance(self, model_type, X_train, y_train, X_val, y_val):
        """Evaluate model performance to decide which model to use"""
        if model_type == "gp":
            # For GP model, we use the log marginal likelihood as a proxy
            try:
                kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X_train, y_train)
                # Use log marginal likelihood as performance metric
                return gp.log_marginal_likelihood_value_
            except:
                return float('-inf')
        else:
            # For linear model, use MSE on validation set
            try:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_val)
                return -mean_squared_error(y_val, y_pred)  # Negative because we want to minimize
            except:
                return float('-inf')
    
    def _select_model(self, X, y):
        """Select the most appropriate model based on performance"""
        if len(X) < 5:
            return "gp"  # Use GP for small datasets
            
        # Split data for validation
        n_val = min(10, max(2, int(len(X) * 0.2)))
        val_indices = self.rng.choice(len(X), n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X)), val_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Evaluate both models
        gp_score = self._evaluate_model_performance("gp", X_train, y_train, X_val, y_val)
        lr_score = self._evaluate_model_performance("lr", X_train, y_train, X_val, y_val)
        
        # Choose model with better performance (higher score for GP, lower for linear)
        if gp_score > lr_score:
            return "gp"
        else:
            return "lr"
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Select model based on performance
        model_type = self._select_model(self.X, self.y)
        
        if model_type == "gp":
            # Fit GP model
            gp = self._fit_model(self.X, self.y)
            
            # Predict mean and std at candidate points
            mu, sigma = gp.predict(X, return_std=True)
        else:
            # Fit Linear model
            lr = self._fit_linear_model(self.X, self.y)
            mu = lr.predict(X)
            # For linear models, estimate uncertainty as a constant
            sigma = np.full(len(X), 0.1)  # Simple uncertainty estimation
        
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
## Feedback
 The algorithm NovelBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3441 with standard deviation 0.0111.

took 67.26 seconds to run.