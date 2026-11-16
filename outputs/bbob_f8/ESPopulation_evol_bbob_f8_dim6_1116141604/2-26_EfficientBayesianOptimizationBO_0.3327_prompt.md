You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on BBOB function 8 (Rosenbrock, original), instance 1, dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EfficientBayesianOptimizationBO: 0.3334, 41.21 seconds, **EfficientBayesianOptimizationBO**: This is a novel Bayesian optimization algorithm that combines multiple strategies to achieve superior performance on black-box optimization problems. Key features include: 1) A hybrid initialization strategy using Latin Hypercube Sampling combined with a simple heuristic for better exploration, 2) An improved acquisition function based on Expected Improvement with a novel adaptive exploration-exploitation balance, 3) A sophisticated selection mechanism that uses a combination of local search and evolutionary strategies to efficiently optimize the acquisition function, 4) Efficient surrogate model management with automatic hyperparameter tuning using a small number of restarts, and 5) Adaptive batch size selection based on remaining budget to maximize information gain.


- EfficientBayesianOptimizationBO: 0.3331, 34.05 seconds, **EfficientBayesianOptimizationBO**: This is a novel Bayesian optimization algorithm that combines multiple strategies to achieve superior performance on black-box optimization problems. Key features include: 1) A hybrid initialization strategy using Latin Hypercube Sampling combined with a simple heuristic for better exploration, 2) An improved acquisition function based on Expected Improvement with a novel adaptive exploration-exploitation balance, 3) A sophisticated selection mechanism that uses a combination of local search and evolutionary strategies to efficiently optimize the acquisition function, 4) Efficient surrogate model management with automatic hyperparameter tuning using a small number of restarts, and 5) Adaptive batch size selection based on remaining budget to maximize information gain.


- EfficientBO: 0.3275, 18.62 seconds, **Efficient Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm enhances the previous EfficientBO by introducing several key improvements:

1. **Smart Budget Allocation**: Dynamically adjusts the number of initial points based on problem characteristics and budget constraints, ensuring better exploration-exploitation balance
2. **Enhanced Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics
3. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration
4. **Multi-Start Optimization**: Implements a more sophisticated multi-start approach using both random sampling and local search techniques
5. **Adaptive Batch Size**: Dynamically adjusts batch size based on remaining budget to maximize information gain
6. **Efficient Model Updates**: Uses selective retraining strategies to minimize computational overhead


- EfficientBO: 0.3262, 16.28 seconds, **Efficient Bayesian Optimization with Adaptive Model Selection and Multi-Start Acquisition**

This novel Bayesian optimization algorithm combines several advanced techniques to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key features include:

1. **Adaptive Model Selection**: Dynamically switches between different surrogate models (Gaussian Process with Matérn kernel and Linear Model) based on problem characteristics
2. **Multi-Start Acquisition Optimization**: Uses multiple restarts with different sampling strategies to efficiently optimize the acquisition function 
3. **Efficient Batch Selection**: Implements a hybrid approach that balances exploration and exploitation using EI acquisition with adaptive parameters
4. **Budget-Aware Strategy**: Adapts the number of initial points and model updates based on available budget




The selected solution to update is:
**Efficient Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm enhances the previous EfficientBO by introducing several key improvements:

1. **Smart Budget Allocation**: Dynamically adjusts the number of initial points based on problem characteristics and budget constraints, ensuring better exploration-exploitation balance
2. **Enhanced Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics
3. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration
4. **Multi-Start Optimization**: Implements a more sophisticated multi-start approach using both random sampling and local search techniques
5. **Adaptive Batch Size**: Dynamically adjusts batch size based on remaining budget to maximize information gain
6. **Efficient Model Updates**: Uses selective retraining strategies to minimize computational overhead


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
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EfficientBO:
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
        
        # Smart budget allocation parameters
        self.adaptive_batch_size = True
        self.model_update_frequency = max(1, budget // 20)
        self.switch_threshold = 0.9  # Switch to linear if linear error < gp_error * threshold

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

    def _acquisition_function(self, X):
        """Compute hybrid acquisition function combining EI and Uncertainty Sampling"""
        if self.model_type == "gp":
            return self._compute_hybrid_acq_gp(X)
        else:
            return self._compute_hybrid_acq_linear(X)

    def _compute_hybrid_acq_gp(self, X):
        """Compute hybrid acquisition for Gaussian Process model"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value
            y_best = np.min(self.y)
            
            # Compute Expected Improvement
            improvement = y_best - mean - 1e-8  # Small epsilon to prevent division by zero
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Compute Uncertainty Sampling (inverse of std)
            uncertainty = 1.0 / (std + 1e-8)
            
            # Combine both for hybrid acquisition
            # Weighting factor: higher weight to EI in later stages, more to uncertainty in early stages
            if self.n_evals > self.budget * 0.5:
                ei_weight = 0.7
                uncertainty_weight = 0.3
            else:
                ei_weight = 0.4
                uncertainty_weight = 0.6
                
            hybrid_acq = ei_weight * ei + uncertainty_weight * uncertainty
            
            return hybrid_acq.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_hybrid_acq_linear(self, X):
        """Compute hybrid acquisition for Linear model"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic (this is not true EI but maintains exploration)
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            # Add uncertainty component based on variance of training data
            y_variance = np.var(self.y) if len(self.y) > 1 else 1.0
            uncertainty = np.ones((X.shape[0], 1)) * y_variance
            
            # Hybrid approach: combine EI with uncertainty
            hybrid_acq = 0.7 * ei + 0.3 * uncertainty
            
            return hybrid_acq
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Use a hybrid approach for acquisition optimization
        best_acq_val = -np.inf
        best_point = None
        
        # Try multiple random starts to avoid local optima
        n_starts = min(30, max(5, self.acq_n_restarts))
        
        # Sample more points for better exploration in early stages
        if self.n_evals < self.budget * 0.3:
            sample_size = 20
        else:
            sample_size = 10
            
        for _ in range(n_starts):
            # Sample initial points for optimization
            start_points = self._sample_points(sample_size)
            
            # Try to optimize acquisition function from these points
            try:
                acq_vals = self._acquisition_function(start_points)
                
                # Select the point with highest acquisition value
                max_idx = np.argmax(acq_vals.flatten())
                if acq_vals[max_idx] > best_acq_val:
                    best_acq_val = acq_vals[max_idx]
                    best_point = start_points[max_idx].reshape(1, -1)
            except:
                continue
        
        if best_point is None:
            # Fallback to random sampling
            return self._sample_points(batch_size)
            
        return best_point

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

    def _should_update_model(self):
        """Determine if we should update the model based on budget and evaluation count"""
        # Update model every few evaluations or when we have enough data
        return (self.n_evals % self.model_update_frequency == 0) and (len(self.X) > 5)

    def _adaptive_batch_size(self):
        """Dynamically adjust batch size based on remaining budget"""
        if self.budget <= 100:
            return 1
        elif self.budget <= 300:
            # For medium budgets, use larger batches in early stages
            if self.n_evals < self.budget * 0.4:
                return min(5, max(1, self.budget // 50))
            else:
                return 1
        else:
            # For large budgets, batch size can be larger initially
            if self.n_evals < self.budget * 0.3:
                return min(10, max(1, self.budget // 40))
            elif self.n_evals < self.budget * 0.7:
                return min(5, max(1, self.budget // 100))
            else:
                return 1

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
                
            # Determine batch size
            batch_size = self._adaptive_batch_size()
            
            # Select next points for evaluation
            X_next = self._select_next_points(batch_size)
            
            # Evaluate next point(s)
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows
            if self._should_update_model() and len(self.X) > 5:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance
            if len(self.X) > 15 and self.n_evals < self.budget * 0.8:
                try:
                    # Test if GP is significantly better than linear
                    if hasattr(self.gp_model, 'predict'):
                        gp_pred = self.gp_model.predict(self.X, return_std=False)
                        if self.linear_model is not None:
                            lin_pred = self.linear_model.predict(self.scaler.transform(self.X)) 
                            if len(lin_pred) == len(self.y):
                                gp_error = mean_squared_error(self.y, gp_pred)
                                lin_error = mean_squared_error(self.y, lin_pred.reshape(-1, 1))
                                
                                # Switch to linear if it's better (but not too early)
                                if lin_error < gp_error * self.switch_threshold and self.model_type != "linear":
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
The algorithm EfficientBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3275 with standard deviation 0.0120.

took 18.62 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

