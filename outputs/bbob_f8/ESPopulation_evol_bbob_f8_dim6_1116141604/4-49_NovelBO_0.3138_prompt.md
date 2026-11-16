You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on BBOB function 8 (Rosenbrock, original), instance 1, dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EfficientBayesianOptimizationBO: 0.3463, 500.50 seconds, **Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm enhances the previous EfficientBO by introducing several key improvements:

1. **Smart Budget Allocation**: Dynamically adjusts the number of initial points based on problem characteristics and budget constraints, ensuring better exploration-exploitation balance
2. **Enhanced Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics
3. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration
4. **Multi-Start Optimization**: Implements a more sophisticated multi-start approach using both random sampling and local search techniques
5. **Adaptive Batch Size**: Dynamically adjusts batch size based on remaining budget to maximize information gain
6. **Efficient Model Updates**: Uses selective retraining strategies to minimize computational overhead


- NovelBayesianOptimizationBO: 0.3432, 42.57 seconds, **Novel Bayesian Optimization with Adaptive Exploration-Exploitation and Efficient Model Updates**

This is a refined Bayesian optimization algorithm that builds upon previous efficient approaches but introduces several novel improvements specifically targeted at black-box optimization problems like BBOB function 8 (Rosenbrock). Key innovations include:

1. **Adaptive Exploration-Exploitation Balance**: The algorithm dynamically adjusts the balance between exploration and exploitation using a sophisticated adaptive factor that considers both iteration progress and problem characteristics
2. **Efficient Model Management**: Implements intelligent surrogate model updates with selective retraining strategies to minimize computational overhead while maintaining accuracy
3. **Enhanced Acquisition Function Optimization**: Uses multiple optimization restarts with hybrid sampling strategies (LHS + local search) for better acquisition function optimization
4. **Smart Budget Allocation**: Dynamically adjusts batch size and initial points based on problem dimensionality and remaining budget to maximize information gain


- EfficientBayesianOptimizationBO: 0.3334, 41.21 seconds, **EfficientBayesianOptimizationBO**: This is a novel Bayesian optimization algorithm that combines multiple strategies to achieve superior performance on black-box optimization problems. Key features include: 1) A hybrid initialization strategy using Latin Hypercube Sampling combined with a simple heuristic for better exploration, 2) An improved acquisition function based on Expected Improvement with a novel adaptive exploration-exploitation balance, 3) A sophisticated selection mechanism that uses a combination of local search and evolutionary strategies to efficiently optimize the acquisition function, 4) Efficient surrogate model management with automatic hyperparameter tuning using a small number of restarts, and 5) Adaptive batch size selection based on remaining budget to maximize information gain.


- NovelBO: 0.3333, 640.31 seconds, **Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm introduces several key improvements over previous versions to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key features include:

1. **Enhanced Adaptive Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics and problem complexity
2. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration-exploitation balance
3. **Multi-Start Acquisition Optimization**: Employs a sophisticated multi-start approach using both random sampling and local search techniques, including trust-region methods
4. **Smart Budget Allocation**: Dynamically adjusts the number of initial points and model updates based on remaining budget to maximize information gain
5. **Efficient Batch Selection**: Implements an adaptive batch size selection strategy that balances exploration and exploitation




The selected solution to update is:
**Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm introduces several key improvements over previous versions to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key features include:

1. **Enhanced Adaptive Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics and problem complexity
2. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration-exploitation balance
3. **Multi-Start Acquisition Optimization**: Employs a sophisticated multi-start approach using both random sampling and local search techniques, including trust-region methods
4. **Smart Budget Allocation**: Dynamically adjusts the number of initial points and model updates based on remaining budget to maximize information gain
5. **Efficient Batch Selection**: Implements an adaptive batch size selection strategy that balances exploration and exploitation


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

class NovelBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Set initial points based on budget and dimension
        if budget <= 50:
            self.n_init = max(5, min(10, budget // 2))
        elif budget <= 200:
            self.n_init = max(10, min(20, budget // 4))
        else:
            self.n_init = max(15, min(30, budget // 6))
        
        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model_type = "gp"  # Switch between 'gp' and 'linear'
        self.gp_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_n_restarts = min(20, max(5, budget // 10))
        self.acq_batch_size = 1
        self.epsilon = 1e-8
        
        # Performance tracking for model switching
        self.gp_performance_history = []
        self.linear_performance_history = []

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling"""
        # Use Sobol sampling for initial points
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        """Fit surrogate model with adaptive selection"""
        if self.model_type == "gp":
            # Fit Gaussian Process with Matern kernel
            try:
                # Use a more robust kernel
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=min(5, max(1, self.n_evals // 10)),
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X, y)
                return gp
            except:
                # Fallback to linear model if GP fails
                self.model_type = "linear"
                return self._fit_linear_model(X, y)
        else:
            return self._fit_linear_model(X, y)

    def _fit_linear_model(self, X, y):
        """Fit a simple linear model as fallback"""
        try:
            # Scale the data for numerical stability
            X_scaled = self.scaler.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_scaled, y.flatten())
            return lr
        except:
            # Fallback to constant prediction
            return np.mean(y)

    def _compute_gp_performance(self, X_test, y_test):
        """Compute performance metric for GP model"""
        if self.gp_model is None:
            return float('inf')
        
        try:
            predictions = self.gp_model.predict(X_test, return_std=False)
            mse = np.mean((predictions.flatten() - y_test.flatten())**2)
            return mse
        except:
            return float('inf')

    def _compute_linear_performance(self, X_test, y_test):
        """Compute performance metric for linear model"""
        if self.linear_model is None:
            return float('inf')
        
        try:
            X_scaled = self.scaler.transform(X_test)
            predictions = self.linear_model.predict(X_scaled)
            mse = np.mean((predictions.flatten() - y_test.flatten())**2)
            return mse
        except:
            return float('inf')

    def _acquisition_function(self, X):
        """Compute Expected Improvement acquisition function with uncertainty sampling"""
        if self.model_type == "gp":
            return self._compute_ei_gp(X)
        else:
            return self._compute_ei_linear(X)

    def _compute_ei_gp(self, X):
        """Compute EI for Gaussian Process model with enhanced exploration"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value
            y_best = np.min(self.y)
            
            # Compute EI with uncertainty sampling
            improvement = y_best - mean - self.epsilon  # Small epsilon to prevent division by zero
            z = improvement / (std + self.epsilon)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Add uncertainty term for exploration
            uncertainty_factor = 0.5 * std
            
            # Combine EI with uncertainty for better exploration-exploitation balance
            combined_acq = ei + uncertainty_factor
            
            return combined_acq.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ei_linear(self, X):
        """Compute EI for Linear model with enhanced exploration"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic (this is not true EI but maintains exploration)
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            # Add uncertainty term for better exploration
            uncertainty_factor = 0.5 * np.std(self.y)  # Simple estimate of uncertainty
            
            return ei + uncertainty_factor
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _optimize_acquisition(self, X_init):
        """Optimize acquisition function using multiple strategies"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(1)
        
        best_acq_val = -np.inf
        best_point = None
        
        # Strategy 1: Local optimization from random starts with trust-region method
        for _ in range(min(5, max(1, self.acq_n_restarts // 2))):
            try:
                # Random start point
                x_start = X_init[self.rng.integers(len(X_init))].reshape(1, -1)
                
                # Use scipy minimize with trust-constr method for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0][0],
                    x_start,
                    method='trust-constr',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'disp': False}
                )
                
                if result.success:
                    acq_val = self._acquisition_function(result.x.reshape(1, -1))[0][0]
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x.reshape(1, -1)
            except:
                continue
        
        # Strategy 2: Global optimization with differential evolution
        if best_point is None:
            try:
                # Use differential evolution for global optimization
                bounds = list(zip(self.bounds[0], self.bounds[1]))
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0][0],
                    bounds,
                    seed=int(self.rng.integers(2**31 - 1)),
                    maxiter=50,
                    popsize=10
                )
                
                if result.success:
                    acq_val = self._acquisition_function(result.x.reshape(1, -1))[0][0]
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x.reshape(1, -1)
            except:
                pass
        
        # Strategy 3: Fallback to sample-based optimization
        if best_point is None:
            # Sample random points and pick the best
            sample_points = self._sample_points(20)
            acq_vals = self._acquisition_function(sample_points)
            max_idx = np.argmax(acq_vals.flatten())
            best_point = sample_points[max_idx].reshape(1, -1)
        
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Generate multiple starting points for optimization
        start_points = self._sample_points(min(50, max(10, self.acq_n_restarts * 2)))
        
        # Optimize acquisition function from these points
        next_point = self._optimize_acquisition(start_points)
        
        return next_point

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate points and return function values
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        # Initialize
        self.X = None
        self.y = None
        self.n_evals = 0
        
        # Generate initial points
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        
        # Store initial points
        self._update_eval_points(X_init, y_init)
        
        # Fit the initial model
        self.gp_model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to continue
            if self.n_evals >= self.budget:
                break
                
            # Select next points for evaluation
            X_next = self._select_next_points(1)
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows
            if len(self.X) > 5 and self.n_evals < self.budget * 0.7:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance
            if len(self.X) > 15 and self.n_evals < self.budget * 0.8:
                try:
                    # Test if GP is significantly better than linear
                    test_indices = self.rng.choice(len(self.X), min(20, len(self.X)), replace=False)
                    X_test = self.X[test_indices]
                    y_test = self.y[test_indices]
                    
                    gp_error = self._compute_gp_performance(X_test, y_test)
                    lin_error = self._compute_linear_performance(X_test, y_test)
                    
                    # Update performance history
                    self.gp_performance_history.append(gp_error)
                    self.linear_performance_history.append(lin_error)
                    
                    # Keep only last 5 evaluations for model selection
                    if len(self.gp_performance_history) > 5:
                        self.gp_performance_history = self.gp_performance_history[-5:]
                        self.linear_performance_history = self.linear_performance_history[-5:]
                    
                    # Switch to linear if it's better (but not too early)
                    if len(self.gp_performance_history) >= 3:
                        gp_avg = np.mean(self.gp_performance_history[-3:])
                        lin_avg = np.mean(self.linear_performance_history[-3:])
                        
                        if lin_avg < gp_avg * 0.85 and self.model_type != "linear":
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                except:
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

```
The algorithm NovelBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3333 with standard deviation 0.0156.

took 640.31 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

