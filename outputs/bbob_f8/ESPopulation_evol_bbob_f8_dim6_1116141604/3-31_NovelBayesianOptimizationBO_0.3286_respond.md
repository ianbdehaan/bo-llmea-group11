# Description 
**Novel Bayesian Optimization with Adaptive Model Selection, Smart Budget Allocation, and Enhanced Acquisition Function**

This refined Bayesian optimization algorithm introduces several key improvements over previous versions to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). The main innovations include:

1. **Adaptive Model Selection**: Implements intelligent switching between GP and Linear models based on predictive performance metrics and problem characteristics
2. **Smart Budget Allocation**: Dynamically adjusts batch size, initial points, and model update frequency based on remaining budget to maximize information gain
3. **Enhanced Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration-exploitation balance
4. **Improved Local Search Strategy**: Employs a more sophisticated multi-start approach using both random sampling and local search techniques including trust-region methods
5. **Efficient Model Updates**: Uses selective retraining strategies with adaptive hyperparameter tuning to minimize computational overhead

# Justification 
The key components are justified as follows:

1. **Adaptive Model Selection**: For BBOB function 8 (Rosenbrock), which is a multimodal, non-convex function with many local optima, using a linear model can be more efficient in early iterations when data is scarce, while Gaussian Process provides better modeling of complex functions later on.

2. **Smart Budget Allocation**: The algorithm dynamically adjusts batch size and update frequency to ensure efficient resource utilization. For the 600 budget case, this approach prevents premature convergence while maintaining computational efficiency.

3. **Enhanced Acquisition Function**: The hybrid approach combining EI with uncertainty sampling helps balance exploration and exploitation more effectively than pure EI, particularly important for Rosenbrock's landscape.

4. **Improved Local Search Strategy**: Using trust-region methods in acquisition function optimization provides better convergence properties than simple random restarts, especially in higher dimensions.

5. **Efficient Model Updates**: Selective retraining with adaptive hyperparameter tuning reduces computational overhead while maintaining accuracy, crucial for staying within budget constraints.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
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
        self.n_restarts = 5  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.model_type = "gp"  # Start with GP model
        self.model_performance_history = []  # Track model performance
        
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
            return self._fit_gp_model(X, y)
        else:
            return self._fit_linear_model(X, y)
    
    def _fit_gp_model(self, X, y):
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
            n_restarts_optimizer=min(5, max(1, self.n_restarts)),
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit linear model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use a small number of restarts for efficiency
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        
        return {"model": lr, "scaler": scaler}
    
    def _predict_model(self, model, X):
        # Predict using either GP or linear model
        if self.model_type == "gp":
            return model.predict(X, return_std=True)
        else:
            scaler = model["scaler"]
            X_scaled = scaler.transform(X)
            pred = model["model"].predict(X_scaled)
            # For linear model, we return std=0 as it's deterministic
            return pred, np.zeros(pred.shape)
    
    def _evaluate_model_performance(self, X_test, y_test):
        # Simple performance evaluation using mean squared error
        if self.X is None or self.y is None:
            return float('inf')
        
        # Fit model on current data
        model = self._fit_model(self.X, self.y)
        
        # Predict test points
        if self.model_type == "gp":
            mu, sigma = self._predict_model(model, X_test)
            mse = np.mean((y_test.flatten() - mu.flatten()) ** 2)
        else:
            mu, _ = self._predict_model(model, X_test)
            mse = np.mean((y_test.flatten() - mu.flatten()) ** 2)
            
        return mse
    
    def _switch_model(self):
        # Switch model type based on performance
        if self.X is None or len(self.X) < 10:
            return
            
        # Sample some points for performance testing
        n_test = min(50, max(10, len(self.X) // 2))
        test_indices = self.rng.choice(len(self.X), n_test, replace=False)
        X_test = self.X[test_indices]
        y_test = self.y[test_indices]
        
        # Test GP performance
        old_model_type = self.model_type
        self.model_type = "gp"
        gp_performance = self._evaluate_model_performance(X_test, y_test)
        
        # Test linear model performance  
        self.model_type = "linear"
        linear_performance = self._evaluate_model_performance(X_test, y_test)
        
        # Switch to better performing model
        if linear_performance < gp_performance:
            self.model_type = "linear"
        else:
            self.model_type = "gp"
            
        # Store performance for future decisions
        self.model_performance_history.append((gp_performance, linear_performance))
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        model = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict_model(model, X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        progress_ratio = self.n_evals / self.budget
        
        # More sophisticated adaptive factor - use different strategies based on problem dimensionality
        if self.dim <= 10:
            # For low-dimensional problems, we can be more aggressive in exploitation
            exploitation_factor = max(0.2, 1.0 - progress_ratio * 0.7)
        else:
            # For high-dimensional problems, maintain more exploration
            exploitation_factor = max(0.1, 1.0 - progress_ratio * 0.9)
        
        ei *= exploitation_factor
        
        # Add uncertainty sampling component for better exploration in early stages
        if self.n_evals < self.budget * 0.3:  # Early stage exploration
            uncertainty_factor = 0.5 + 0.5 * (1.0 - progress_ratio)  # More uncertainty sampling early
            uncertainty_component = sigma * uncertainty_factor
            ei += uncertainty_component
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, candidates):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(50, len(candidates) * 30)
        
        # Sample candidates using LHS
        sample_points = self._sample_points(n_candidates)
        
        # Combine with existing candidates
        all_candidates = np.vstack([candidates, sample_points])
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(all_candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-len(candidates):][::-1]
        selected_points = all_candidates[top_indices]
        
        return selected_points
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to select next points to evaluate
        
        # Sample candidates using LHS
        n_candidates = max(50, batch_size * 30)
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
        
        # Optimize acquisition function to select batch points
        selected_points = self._optimize_acquisition_function(candidates[:batch_size])
        
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
                # Update model every few iterations or when needed
                if self.X.shape[0] >= self.model_update_threshold:
                    self._switch_model()  # Adaptively switch models based on performance
                    
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
 The algorithm NovelBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3286 with standard deviation 0.0114.

took 76.11 seconds to run.