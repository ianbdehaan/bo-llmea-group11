You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on the real-world problem 1124 (parameter estimation of Frequency Modulated Sound Waves), dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EGOABLBO: 0.4485, 9.41 seconds, Efficient Global Optimization with Adaptive Bayesian Learning (EGOABL) is a novel Bayesian optimization algorithm designed for black-box optimization problems. It combines adaptive sampling strategies, efficient Gaussian process modeling, and smart acquisition function selection to achieve superior performance within limited budgets. The method uses a hybrid approach where it starts with Latin Hypercube Sampling for diverse initial exploration, then switches to Thompson Sampling for exploitation, and dynamically adjusts the balance between exploration and exploitation using an adaptive acquisition function that combines Expected Improvement with Uncertainty Quantification.


- EfficientBO: 0.4482, 10.10 seconds, "Efficient Bayesian Optimization with Adaptive Sampling and Multi-Start Acquisition"

This novel Bayesian optimization algorithm combines several efficiency-enhancing strategies:

1. **Adaptive Initial Sampling**: Uses Latin Hypercube Sampling (LHS) for the initial points, but dynamically adjusts the number of initial points based on dimensionality and budget to ensure good coverage while respecting budget constraints.

2. **Multi-Start Acquisition Optimization**: Instead of using a single optimization method for acquisition function maximization, it employs multiple random starts with different strategies (random, Sobol, LHS) to find better acquisition maxima, improving exploration-exploitation balance.

3. **Efficient GP Model Updates**: Uses a sparse GP model with automatic relevance detection (ARD) kernel and avoids expensive hyperparameter optimization by using fixed lengthscales and optimizing only the signal variance and noise variance.

4. **Adaptive Batch Selection**: Implements a batch selection strategy that considers both acquisition value and diversity to prevent getting stuck in local maxima while maintaining computational efficiency.

5. **Budget-aware Operations**: The algorithm is designed to work efficiently across different budgets by dynamically adjusting internal parameters and avoiding expensive operations as the budget decreases.


- EfficientBO: 0.0000, 0.00 seconds, "Efficient Bayesian Optimization with Adaptive Sampling and Multi-Start Acquisition Maximization"


- EMOBO: 0.0000, 0.00 seconds, "Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Gradient-Based Local Search (EMOBO)"

This algorithm combines several key innovations to achieve superior performance in black-box optimization with limited budget. It uses a multi-start approach where multiple initial designs are generated using Latin Hypercube Sampling, followed by a novel acquisition function that balances exploration and exploitation more effectively than standard UCB or EI. The algorithm employs adaptive sampling strategies that increase the number of candidate points for selection based on the optimization progress to ensure sufficient exploration early on, while focusing on exploitation later. A key innovation is the inclusion of gradient-based local search at selected points to accelerate convergence, which is only performed when beneficial and within budget constraints.




The selected solution to update is:
"Efficient Bayesian Optimization with Adaptive Sampling and Multi-Start Acquisition Maximization"


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
import warnings

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
        self.n_init = max(5, min(20, dim * 3))  # Adaptive initial points based on dimensionality
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model
        self.model = None
        
        # Acquisition function parameters
        self.noise_var = 1e-6  # Small noise variance for numerical stability

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a Matern kernel with automatic relevance detection (ARD)
        # Add WhiteKernel for noise modeling
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(
            noise_level=self.noise_var, noise_level_bounds=(1e-10, 1e+1)
        )
        
        # Use a sparse GP for computational efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # No additional regularization
            normalize_y=True,
            n_restarts_optimizer=2,  # Few restarts to keep cost low
            random_state=int(self.rng.integers(2**31 - 1)),
            noise="gaussian"
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI) acquisition function
        # This is a computationally efficient implementation
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Get predictions from the model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Get the best observed value so far
        y_best = np.min(self.y) if len(self.y) > 0 else 0.0
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        # Compute EI
        improvement = y_best - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using multi-start optimization
        
        # Start with some random points for exploration
        n_random_starts = max(5, min(20, batch_size * 3))
        
        # Generate multiple random starting points
        random_starts = self._sample_points(n_random_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Try different optimization strategies with multiple starting points
        for i in range(n_random_starts):
            start_point = random_starts[i:i+1]
            
            # Try L-BFGS with the current start point
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point.flatten(),
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    candidate = result.x.reshape(1, -1)
                    value = self._acquisition_function(candidate)[0, 0]
                    if value > best_value:
                        best_value = value
                        best_point = candidate
            except:
                pass
                
            # Also try Nelder-Mead for robustness
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 20}
                )
                
                if result.success:
                    candidate = result.x.reshape(1, -1)
                    value = self._acquisition_function(candidate)[0, 0]
                    if value > best_value:
                        best_value = value
                        best_point = candidate
            except:
                pass
        
        # If we didn't find a good point with optimization, return a random one
        if best_point is None:
            best_point = self._sample_points(1)
            
        return np.tile(best_point, (batch_size, 1))

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception:
                # If evaluation fails, use a large value to penalize
                y_vals.append(1e6)
                
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)
    
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
        
        # Initialize with some random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Determine how many points to evaluate in this batch
            remaining_budget = self.budget - self.n_evals
            
            # For the last few evaluations, reduce batch size
            if remaining_budget <= 10:
                batch_size = 1
            elif remaining_budget <= 50:
                batch_size = min(5, remaining_budget)
            else:
                # Use a reasonable batch size (1-3 for efficiency)
                batch_size = min(max(1, int(remaining_budget / 100)), 3)
            
            # If we can't evaluate anything, break
            if batch_size <= 0 or self.n_evals >= self.budget:
                break
                
            # Select next points using acquisition function
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the model with new data
            self._update_eval_points(next_points, next_values)
            
            # Refit the model occasionally (every 5 iterations or so)
            if self.n_evals % 5 == 0:
                try:
                    self.model = self._fit_model(self.X, self.y)
                except Exception as e:
                    # If refitting fails, continue with old model
                    warnings.warn(f"Model refitting failed: {e}")

        # Return the best point found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()

```
An error occurred : Traceback (most recent call last):
  File "<EfficientBO>", line 179, in __call__
 179->         self.model = self._fit_model(self.X, self.y)
  File "<EfficientBO>", line 48, in _fit_model
  46 |         
  47 |         # Use a sparse GP for computational efficiency
  48->         model = GaussianProcessRegressor(
  49 |             kernel=kernel,
  50 |             alpha=0.0,  # No additional regularization
TypeError: GaussianProcessRegressor.__init__() got an unexpected keyword argument 'noise'


Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

