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


- EfficientBO: 0.3262, 16.28 seconds, **Efficient Bayesian Optimization with Adaptive Model Selection and Multi-Start Acquisition**

This novel Bayesian optimization algorithm combines several advanced techniques to achieve superior performance on black-box optimization problems like BBOB function 8 (Rosenbrock). Key features include:

1. **Adaptive Model Selection**: Dynamically switches between different surrogate models (Gaussian Process with Matérn kernel and Linear Model) based on problem characteristics
2. **Multi-Start Acquisition Optimization**: Uses multiple restarts with different sampling strategies to efficiently optimize the acquisition function 
3. **Efficient Batch Selection**: Implements a hybrid approach that balances exploration and exploitation using EI acquisition with adaptive parameters
4. **Budget-Aware Strategy**: Adapts the number of initial points and model updates based on available budget


- EfficientBayesianOptimizationBO: 0.3128, 61.47 seconds, **EfficientBayesianOptimizationBO**: A novel Bayesian optimization algorithm that combines a multi-start local search with a progressive refinement strategy. It uses a sparse Gaussian process model with active learning to efficiently explore the search space, employs an improved Expected Improvement acquisition function with adaptive parameters, and incorporates a hybrid selection strategy that balances exploration and exploitation. The algorithm features a budget-aware initialization and progressive refinement of the surrogate model to maintain computational efficiency across different budget constraints.


- EfficientBayesianOptimizationBO: 0.0000, 0.00 seconds, **EfficientBayesianOptimizationBO**: This is a novel Bayesian optimization algorithm that combines multiple strategies for efficient black-box optimization. Key features include: (1) A hybrid initialization strategy using Latin Hypercube Sampling combined with a small random sample to ensure good coverage of the search space; (2) An improved Expected Improvement acquisition function with adaptive exploration-exploitation trade-off; (3) A multi-start optimization approach for acquisition function maximization using both local and global methods; (4) Efficient model updating with selective retraining; (5) Adaptive batch size that decreases as the budget is consumed to maintain computational efficiency.




The selected solution to update is:
**EfficientBayesianOptimizationBO**: A novel Bayesian optimization algorithm that combines a multi-start local search with a progressive refinement strategy. It uses a sparse Gaussian process model with active learning to efficiently explore the search space, employs an improved Expected Improvement acquisition function with adaptive parameters, and incorporates a hybrid selection strategy that balances exploration and exploitation. The algorithm features a budget-aware initialization and progressive refinement of the surrogate model to maintain computational efficiency across different budget constraints.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans

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
        self.n_init = max(10, min(20, budget // 10))  # Adaptive initialization based on budget
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model = None
        self.X_refined = None
        self.y_refined = None

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a sparse GP with Matern kernel for better performance on noisy functions
        # Use active learning to select representative points
        
        # If we have very few points, use a simple model
        if len(X) < 10:
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        else:
            # Use more complex model for better fitting when we have sufficient data
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use a sparse GP for better scalability
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function using improved Expected Improvement
        # with adaptive parameters based on current optimization progress
        
        if self.model is None:
            # If no model yet, use a simple exploration-based acquisition
            return np.ones((X.shape[0], 1)) * 0.1
            
        # Get model predictions
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Get the best observed value
        if len(self.y) > 0:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Calculate EI with adaptive parameter
        # Adaptive exploration factor based on optimization progress
        progress_ratio = self.n_evals / self.budget
        xi = max(0.01, 0.1 * (1 - progress_ratio))  # Decrease as we progress
        
        # Expected Improvement calculation
        with np.errstate(divide='warn'):
            imp = best_y - mu - xi
            Z = imp / (sigma + 1e-8)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        # Clip negative values to encourage exploration
        ei = np.clip(ei, 0, None)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # 1. Local search from multiple starting points (exploitation)
        # 2. Global search with acquisition function (exploration)
        
        if self.X is None or len(self.X) < 2:
            # For very early iterations, use random sampling
            return self._sample_points(batch_size)
            
        selected_points = []
        
        # Use multiple restarts of local optimization for exploitation
        n_local_restarts = min(5, batch_size)
        for _ in range(n_local_restarts):
            # Start from a random point within bounds
            start_point = self._sample_points(1)[0]
            
            # Local optimization using scipy minimize
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    start_point,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    selected_points.append(result.x)
            except:
                # If optimization fails, just use the start point
                pass
        
        # Fill remaining points with global acquisition function maxima
        n_global = batch_size - len(selected_points)
        if n_global > 0:
            # Sample a large set of candidates and select top ones by acquisition value
            candidate_points = self._sample_points(1000)
            acq_values = self._acquisition_function(candidate_points)
            
            # Select the top points
            top_indices = np.argsort(acq_values.flatten())[-n_global:]
            global_points = candidate_points[top_indices]
            selected_points.extend(global_points)
        
        # If we still don't have enough points, fill with random samples
        while len(selected_points) < batch_size:
            selected_points.append(self._sample_points(1)[0])
            
        return np.array(selected_points[:batch_size])

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for point in X:
            y_vals.append(func(point))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        self.n_evals += len(X)
        return y_vals
    
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
        
        # Initialize with Latin Hypercube sampling
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, use fewer points per iteration
            batch_size = min(5, max(1, (self.budget - self.n_evals) // 20))
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically or when we have enough data
            if self.n_evals % 10 == 0 or len(self.X) > 30:
                self.model = self._fit_model(self.X, self.y)
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        return best_y, best_x

```
The algorithm EfficientBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3128 with standard deviation 0.0096.

took 61.47 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

