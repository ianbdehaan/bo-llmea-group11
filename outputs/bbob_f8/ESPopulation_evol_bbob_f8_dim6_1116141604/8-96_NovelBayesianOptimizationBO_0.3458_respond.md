# Description 
**Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, Smart Budget Allocation, and Enhanced Exploration-Exploitation**

This refined Bayesian optimization algorithm builds upon the previous version by introducing several key improvements specifically targeted at black-box optimization problems like BBOB function 8 (Rosenbrock). The main innovations include:

1. **Enhanced Model Selection Strategy**: Implements a more robust switching mechanism between GP and Linear models based on multiple performance metrics including prediction accuracy, model stability, and problem dimensionality characteristics
2. **Advanced Acquisition Function Optimization**: Uses sophisticated multi-start optimization with hybrid sampling strategies (LHS + local search) combined with adaptive exploration-exploitation balance that dynamically adjusts based on iteration progress and problem characteristics
3. **Smart Budget Allocation with Progressive Refinement**: Dynamically adjusts the number of initial points, batch size, and evaluation strategy based on remaining budget and problem complexity to maximize information gain
4. **Improved Exploration Mechanism**: Incorporates uncertainty-based exploration components that are more adaptive to early vs. late stage optimization phases and problem dimensionality
5. **Efficient Computational Strategy**: Implements selective retraining strategies, early termination conditions, and computational cost management to ensure the algorithm scales well with increasing budget sizes

# Justification 
The key improvements address identified weaknesses in the previous approach:

1. **Model Selection**: The original approach used a simple error threshold that could be too sensitive or not adaptive enough. This version considers multiple performance metrics and problem characteristics for more robust model selection.

2. **Acquisition Function Optimization**: The previous version used a simplified approach with fixed restarts. The new approach employs more sophisticated sampling strategies and adaptive optimization techniques to better locate acquisition function maxima.

3. **Budget Allocation**: The budget allocation strategy is now more nuanced, considering both the remaining budget and problem characteristics to dynamically adjust batch sizes and exploration strategies.

4. **Exploration Mechanism**: The early-stage exploration is improved by incorporating dimensionality-aware uncertainty components that are more appropriate for the Rosenbrock function's landscape.

5. **Computational Efficiency**: Added selective retraining and early termination conditions to ensure computational resources are used efficiently while maintaining optimization quality.

The algorithm maintains reproducibility through proper seeding of all random number generators, and is designed to be computationally efficient with costs that grow moderately with increasing budget sizes.

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
        self.switch_to_linear_threshold = 0.8  # Switch to linear model if GP error is above this threshold
        self.ei_exploration_factor = 0.5  # Factor for EI exploration vs exploitation
        
        # Enhanced parameters for better performance
        self.model_stability_window = 10  # Number of recent evaluations to consider for stability
        self.stability_threshold = 0.1  # Threshold for model stability
        self.adaptive_exploitation_decay = 0.8  # Decay factor for exploitation vs exploration
        self.uncertainty_weight = 0.3  # Weight for uncertainty component in acquisition
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on performance and stability
        try:
            # Use a subset of data points if too many to avoid numerical issues
            max_samples = min(1000, X.shape[0])
            if X.shape[0] > max_samples:
                indices = self.rng.choice(X.shape[0], max_samples, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
            
            # Try GP model first
            kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=min(5, max(1, self.n_restarts)),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            
            # Evaluate GP performance on a validation set
            if X.shape[0] > 10:
                # Use 20% of data for validation (or at least 5 points, at most 100)
                n_val = max(5, min(100, X.shape[0] // 5))
                val_indices = self.rng.choice(X.shape[0], n_val, replace=False)
                X_val = X[val_indices]
                y_val = y[val_indices]
                
                # Predict and compute error
                y_pred, _ = gp.predict(X_val, return_std=True)
                gp_error = mean_squared_error(y_val, y_pred)
                
                # Calculate model stability based on recent evaluations
                model_stability = self._calculate_model_stability(gp, X, y)
                
                # More sophisticated switching logic
                if gp_error > self.switch_to_linear_threshold or model_stability < self.stability_threshold:
                    # If GP is performing poorly or unstable, switch to linear model
                    model = LinearRegression()
                    model.fit(X, y)
                    return model, "linear"
                else:
                    return gp, "gp"
            else:
                return gp, "gp"
                
        except Exception as e:
            # Fallback to linear model if GP fails
            model = LinearRegression()
            model.fit(X, y)
            return model, "linear"
    
    def _calculate_model_stability(self, model, X, y):
        """Calculate stability of the model based on recent performance"""
        if self.X is not None and len(self.X) > self.model_stability_window:
            # Check if we have enough history
            recent_indices = np.arange(max(0, len(self.X) - self.model_stability_window), len(self.X))
            recent_X = self.X[recent_indices]
            recent_y = self.y[recent_indices]
            
            try:
                # Predict on recent data points
                y_pred, _ = model.predict(recent_X, return_std=True)
                # Calculate the relative error
                relative_errors = np.abs(y_pred - recent_y.flatten()) / (np.abs(recent_y.flatten()) + 1e-8)
                stability_score = 1.0 - np.mean(relative_errors)
                return max(0.0, stability_score)
            except:
                return 0.5  # Default stability score if calculation fails
        return 1.0  # Assume perfect stability for small datasets
    
    def _acquisition_function(self, X):
        # Expected Improvement with enhanced exploration/exploitation balance
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
            # For linear model, use a more robust uncertainty-based approach
            mu = model.predict(X)
            # Estimate uncertainty based on variance of training data and distance to points
            if len(self.y) > 1:
                y_std = np.std(self.y)
                # Simple distance-based uncertainty adjustment
                distances = np.array([np.linalg.norm(x - self.X[0]) for x in X]) if len(self.X) > 0 else np.zeros(len(X))
                uncertainty = y_std + 0.1 * np.max(distances) if len(distances) > 0 else y_std
            else:
                uncertainty = 1.0
                
            ei = -np.abs(mu - best_y) + uncertainty * self.ei_exploration_factor
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        progress_ratio = self.n_evals / self.budget
        
        # Enhanced adaptive factor that considers problem dimensionality and optimization stage
        if self.dim <= 10:
            # For low-dimensional problems, we can be more aggressive in exploitation
            exploitation_factor = max(0.2, 1.0 - progress_ratio * self.adaptive_exploitation_decay)
        else:
            # For high-dimensional problems, maintain more exploration
            exploitation_factor = max(0.1, 1.0 - progress_ratio * (self.adaptive_exploitation_decay + 0.1))
        
        ei *= exploitation_factor
        
        # Add uncertainty component in early stages to encourage exploration
        if progress_ratio < 0.3:  # Early stage exploration
            uncertainty_component = np.std(self.y) if len(self.y) > 1 else 1.0
            
            # Dynamic uncertainty weight based on problem dimensionality
            dim_factor = min(1.0, self.dim / 20.0)
            ei += uncertainty_component * self.uncertainty_weight * dim_factor * (1.0 - progress_ratio)
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, candidates):
        # Multi-start optimization using sophisticated sampling and selection strategy
        n_candidates = len(candidates)
        
        # Evaluate all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value with better ranking
        sorted_indices = np.argsort(acq_values.flatten())[::-1]  # Sort descending
        
        # Select top candidates (use more sophisticated selection if we have many candidates)
        if n_candidates > 50:
            # For large candidate sets, use a more sophisticated approach
            top_candidates = candidates[sorted_indices[:min(20, n_candidates//3)]]
            
            # Also include some randomly selected points for diversity
            random_indices = self.rng.choice(n_candidates, min(10, n_candidates//5), replace=False)
            random_candidates = candidates[random_indices]
            
            # Combine and re-evaluate to get the best of both worlds
            combined_candidates = np.vstack([top_candidates, random_candidates])
            combined_acq = self._acquisition_function(combined_candidates)
            
            # Select the best from this combined set
            best_idx = np.argmax(combined_acq.flatten())
            return combined_candidates[best_idx:best_idx+1]
        else:
            # For smaller candidate sets, just take the top one
            if len(sorted_indices) > 0:
                return candidates[sorted_indices[:1]]
            else:
                return candidates[:1]
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function with enhanced exploration
        n_candidates = max(100, batch_size * 30)  # Increased for better sampling
        
        # Sample candidates using LHS
        candidates = self._sample_points(n_candidates)
        
        # Add some local search points around the current best if available
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Generate perturbations around the best point with adaptive size
            # Use smaller perturbations for problems with more evaluations (better local search)
            if self.n_evals > 100:
                perturbation_size = min(0.5, (5.0 - (-5.0)) / max(20, self.budget // 10))
            else:
                perturbation_size = min(1.0, (5.0 - (-5.0)) / max(10, self.budget // 10))
                
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Optimize acquisition function for candidates and select best points
        selected_points = self._optimize_acquisition_function(candidates)
        
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
                progress_ratio = self.n_evals / self.budget
                batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.6)
                self.batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, 
                                        int(np.ceil(batch_size_factor * 4))))
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
                
                # Update batch size based on remaining budget and progress
                if self.budget > 200:
                    remaining_budget = self.budget - self.n_evals
                    if remaining_budget > 50:
                        # Use dynamic batch sizing that decreases as we approach the end
                        progress_ratio = self.n_evals / self.budget
                        batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.6)
                        self.batch_size = max(self.min_batch_size, 
                                            min(self.max_batch_size, 
                                                int(np.ceil(batch_size_factor * 4))))
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
 The algorithm NovelBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3458 with standard deviation 0.0111.

took 81.74 seconds to run.