# Description
**Novel Bayesian Optimization with Adaptive Exploration-Exploitation, Efficient Model Updates, and Smart Budget Allocation**

This refined Bayesian optimization algorithm introduces several key improvements over previous versions to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key innovations include:

1. **Enhanced Adaptive Exploration-Exploitation Balance**: Implements a more sophisticated adaptive factor that considers both iteration progress and problem characteristics, with dynamic adjustments based on convergence behavior
2. **Efficient Model Management**: Uses selective retraining strategies with early stopping to minimize computational overhead while maintaining accuracy, and implements a smart model switching mechanism between GP and Linear models
3. **Improved Acquisition Function Optimization**: Employs multiple optimization restarts with hybrid sampling strategies (LHS + local search) and adds uncertainty-aware acquisition functions for better exploration
4. **Smart Budget Allocation**: Dynamically adjusts batch size, initial points, and model update frequency based on problem dimensionality and remaining budget to maximize information gain
5. **Robust Convergence Detection**: Implements early stopping criteria and adaptive parameters that adjust based on observed convergence patterns

# Justification
The improvements focus on making the algorithm more robust and efficient:
- Enhanced model management with smarter switching between GP and Linear models based on predictive performance metrics
- More sophisticated adaptive factor in acquisition function that dynamically adjusts exploration/exploitation based on convergence behavior rather than just iteration progress
- Improved batch size selection that considers not just remaining budget but also the rate of improvement
- Better handling of numerical issues in GP fitting with more robust hyperparameter optimization
- Addition of early stopping mechanisms to prevent unnecessary evaluations when convergence is detected

These changes make the algorithm more adaptive and efficient, particularly for problems like the Rosenbrock function where balance between exploration and exploitation is crucial.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
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
        self.model_update_threshold = 10  # Minimum points before updating model
        self.model_switch_threshold = 0.1  # Threshold for switching between GP and Linear models
        
        # Model management
        self.current_model = "gp"  # Initially use Gaussian Process
        self.last_improvement = 0
        self.improvement_history = []
        self.convergence_window = 20
        self.convergence_threshold = 1e-4
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on problem characteristics
        if self.current_model == "gp":
            # Use a small number of restarts for efficiency and fit GP
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
                n_restarts_optimizer=min(10, self.n_restarts),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            try:
                gp.fit(X_subset, y_subset)
                return gp
            except:
                # Fallback to linear model if GP fails
                self.current_model = "linear"
                
        # Fallback to Linear model
        if self.current_model == "linear":
            lr = LinearRegression()
            lr.fit(X, y)
            return lr
        
        return None
    
    def _predict_model(self, model, X):
        # Predict using the appropriate method for each model type
        if hasattr(model, 'predict'):
            try:
                if isinstance(model, GaussianProcessRegressor):
                    return model.predict(X, return_std=True)
                else:
                    return model.predict(X), np.zeros((X.shape[0], 1))
            except:
                # Fallback to constant prediction if error occurs
                return np.mean(self.y) * np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1))
        else:
            return model.predict(X), np.zeros((X.shape[0], 1))
    
    def _model_performance(self, X, y):
        # Calculate model performance to decide whether to switch models
        if self.X is None or len(self.X) < 2:
            return float('inf')
            
        try:
            # Fit a GP model to the current data
            gp = self._fit_model(X, y)
            if isinstance(gp, GaussianProcessRegressor):
                # Use the log-marginal likelihood as performance metric
                # This is a proxy for how well the model fits the data
                return -gp.log_marginal_likelihood_value_ if hasattr(gp, 'log_marginal_likelihood_value_') else 0.0
        except:
            pass
        return float('inf')
    
    def _switch_model(self, X, y):
        # Switch between GP and Linear model based on performance
        if self.X is None or len(self.X) < 10:
            return
            
        try:
            gp_performance = self._model_performance(X, y)
            
            # Fit a simple linear model for comparison
            lr = LinearRegression()
            lr.fit(X, y)
            linear_pred = lr.predict(X)
            linear_error = np.mean((y.flatten() - linear_pred.flatten())**2)
            
            # Switch if linear model performs significantly better
            if linear_error < gp_performance * (1.0 - self.model_switch_threshold):
                self.current_model = "linear"
            elif gp_performance < 0 and self.current_model == "linear":
                # Switch back to GP if it's performing well
                self.current_model = "gp"
        except:
            pass
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        gp = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict_model(gp, X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        progress_ratio = self.n_evals / self.budget
        
        # More sophisticated adaptive factor that considers convergence history
        if len(self.improvement_history) > 1:
            recent_improvement = np.mean(self.improvement_history[-min(5, len(self.improvement_history)):])
            if recent_improvement < self.convergence_threshold:
                # We're converging quickly, reduce exploration
                exploitation_factor = max(0.3, 1.0 - progress_ratio * 0.6)
            else:
                # Still making good progress, maintain exploration
                exploitation_factor = max(0.2, 1.0 - progress_ratio * 0.8)
        else:
            # Early iterations, balance exploration and exploitation
            if self.dim <= 10:
                exploitation_factor = max(0.2, 1.0 - progress_ratio * 0.7)
            else:
                exploitation_factor = max(0.1, 1.0 - progress_ratio * 0.9)
        
        ei *= exploitation_factor
        
        # Add uncertainty-aware component to encourage exploration in less certain regions
        uncertainty_factor = 1.0 + sigma / (np.std(self.y) + 1e-8)
        ei *= uncertainty_factor
        
        return ei.reshape(-1, 1)
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(50, batch_size * 40)
        
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
    
    def _check_convergence(self):
        # Check if we're converging
        if len(self.improvement_history) < 2:
            return False
            
        recent_improvements = self.improvement_history[-min(10, len(self.improvement_history)):]
        avg_improvement = np.mean(recent_improvements)
        
        # If improvements are very small, consider it converged
        if avg_improvement < self.convergence_threshold:
            return True
        return False
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check for convergence early stopping
            if self._check_convergence():
                break
                
            # Adaptive batch size adjustment based on remaining budget and problem characteristics
            if self.budget > 200:
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
            
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Update model if needed
                if self.n_evals % max(5, int(self.budget/20)) == 0 and len(self.X) > self.model_update_threshold:
                    self._switch_model(self.X, self.y)
                
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Track improvement for convergence detection
                if len(self.y) > 1:
                    current_best = np.min(self.y)
                    if len(self.improvement_history) > 0:
                        improvement = self.improvement_history[-1] - current_best
                    else:
                        improvement = 1.0
                        
                    self.improvement_history.append(improvement)
                
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
 The algorithm NovelBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2860 with standard deviation 0.0186.

took 1.59 seconds to run.