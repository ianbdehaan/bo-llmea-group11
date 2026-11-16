You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on the real-world problem 1124 (parameter estimation of Frequency Modulated Sound Waves), dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EGOABLBO: 0.4495, 6.20 seconds, Efficient Global Optimization with Adaptive Bayesian Learning (EGOABL) is a novel Bayesian optimization algorithm designed for black-box optimization problems. It combines adaptive sampling strategies, efficient Gaussian process modeling, and smart acquisition function selection to achieve superior performance within limited budgets. The method uses a hybrid approach where it starts with Latin Hypercube Sampling for diverse initial exploration, then switches to Thompson Sampling for exploitation, and dynamically adjusts the balance between exploration and exploitation using an adaptive acquisition function that combines Expected Improvement with Uncertainty Quantification.


- AEBO: 0.4490, 6.77 seconds, Novel Bayesian Optimization Algorithm with Adaptive Exploration-Exploitation and Efficient Gaussian Process Modeling (AEBO). This algorithm combines: 1) Dynamic exploration-exploitation balance using a hybrid acquisition function that transitions from pure exploration to exploitation, 2) Efficient sparse Gaussian process modeling with automatic relevance detection and fixed hyperparameters for computational efficiency, 3) Multi-start optimization of acquisition function with adaptive restart strategies, 4) Budget-aware batch selection that scales with remaining budget, 5) Adaptive initial sampling size based on dimensionality and budget.


- EGOABLBO: 0.4485, 9.41 seconds, Efficient Global Optimization with Adaptive Bayesian Learning (EGOABL) is a novel Bayesian optimization algorithm designed for black-box optimization problems. It combines adaptive sampling strategies, efficient Gaussian process modeling, and smart acquisition function selection to achieve superior performance within limited budgets. The method uses a hybrid approach where it starts with Latin Hypercube Sampling for diverse initial exploration, then switches to Thompson Sampling for exploitation, and dynamically adjusts the balance between exploration and exploitation using an adaptive acquisition function that combines Expected Improvement with Uncertainty Quantification.


- EfficientBO: 0.4482, 10.10 seconds, "Efficient Bayesian Optimization with Adaptive Sampling and Multi-Start Acquisition"

This novel Bayesian optimization algorithm combines several efficiency-enhancing strategies:

1. **Adaptive Initial Sampling**: Uses Latin Hypercube Sampling (LHS) for the initial points, but dynamically adjusts the number of initial points based on dimensionality and budget to ensure good coverage while respecting budget constraints.

2. **Multi-Start Acquisition Optimization**: Instead of using a single optimization method for acquisition function maximization, it employs multiple random starts with different strategies (random, Sobol, LHS) to find better acquisition maxima, improving exploration-exploitation balance.

3. **Efficient GP Model Updates**: Uses a sparse GP model with automatic relevance detection (ARD) kernel and avoids expensive hyperparameter optimization by using fixed lengthscales and optimizing only the signal variance and noise variance.

4. **Adaptive Batch Selection**: Implements a batch selection strategy that considers both acquisition value and diversity to prevent getting stuck in local maxima while maintaining computational efficiency.

5. **Budget-aware Operations**: The algorithm is designed to work efficiently across different budgets by dynamically adjusting internal parameters and avoiding expensive operations as the budget decreases.




The selected solution to update is:
Efficient Global Optimization with Adaptive Bayesian Learning (EGOABL) is a novel Bayesian optimization algorithm designed for black-box optimization problems. It combines adaptive sampling strategies, efficient Gaussian process modeling, and smart acquisition function selection to achieve superior performance within limited budgets. The method uses a hybrid approach where it starts with Latin Hypercube Sampling for diverse initial exploration, then switches to Thompson Sampling for exploitation, and dynamically adjusts the balance between exploration and exploitation using an adaptive acquisition function that combines Expected Improvement with Uncertainty Quantification.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

class EGOABLBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, dim * 2))  # Adaptive initial points based on dimensionality

        # Do not add any other arguments without a default value
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None
        self.is_initial_phase = True
        self.acquisition_batch_size = 1  # Start with small batches

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for diverse initial exploration
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a sparse GP with Matern kernel for efficiency
        # We'll use a simple approach with cross-validation to select the best hyperparameters
        
        # Create a base kernel (Matern with automatic relevance determination)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        # For computational efficiency, use a subset of points for fitting (especially important when budget is small)
        max_points_for_fit = min(100, X.shape[0])
        fit_indices = self.rng.choice(X.shape[0], size=max_points_for_fit, replace=False)
        X_fit = X[fit_indices]
        y_fit = y[fit_indices]
        
        # Fit the GP model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,  # Reduced number of restarts for efficiency
            random_state=int(self.rng.integers(2**31 - 1)),
            normalize_y=True
        )
        
        gp.fit(X_fit, y_fit)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function using Expected Improvement with uncertainty scaling
        # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
        
        if self.model is None:
            # If no model yet, return random acquisition values for exploration
            return np.ones((X.shape[0], 1)) * self.rng.random(X.shape[0]).reshape(-1, 1)
            
        # Get predictions from the model
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Find current best value (excluding noise)
        if self.y is not None and len(self.y) > 0:
            f_best = np.min(self.y)
        else:
            f_best = 0.0
            
        # Calculate EI using the formula
        improvement = f_best - mu
        with np.errstate(divide='warn'):
            Z = improvement / (sigma + 1e-8)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        # Add uncertainty scaling factor to encourage exploration in early stages
        if self.is_initial_phase:
            # Increase acquisition values for exploration in early phase
            ei = ei * (1.0 + 0.5 * np.exp(-self.n_evals / max(1, self.budget * 0.3)))
            
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # - Use random search in early phase for exploration
        # - Switch to acquisition function optimization for exploitation
        
        if self.is_initial_phase or self.n_evals < 10:
            # Use random sampling during initial phase
            return self._sample_points(batch_size)
        
        # For more advanced selection, we'll optimize the acquisition function
        # Use a simple gradient-free approach with multiple restarts
        
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple starting points for optimization
        n_restarts = min(10, max(3, self.budget // 20))
        
        for _ in range(n_restarts):
            # Start from a random point within bounds
            x0 = self._sample_points(1)[0]
            
            # Simple local search (gradient-free)
            # We'll use a simple optimization approach with few iterations
            
            # Sample several points around the current best to find better candidates
            candidates = self._sample_points(20)
            acq_vals = self._acquisition_function(candidates)
            
            # Select the best candidate from these samples
            max_idx = np.argmax(acq_vals.flatten())
            candidate_x = candidates[max_idx]
            
            if acq_vals[max_idx] > best_acq_val:
                best_acq_val = acq_vals[max_idx]
                best_x = candidate_x.reshape(1, -1)
        
        # If no good candidates found, fall back to random
        if best_x is None:
            return self._sample_points(batch_size)
            
        return best_x

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        if X is None or len(X) == 0:
            return np.array([]).reshape(0, 1)
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception as e:
                # If function evaluation fails, use a large value to penalize
                print(f"Function evaluation failed: {e}")
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Fit the initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we should switch to exploitation phase
            if self.n_evals > self.budget * 0.4:
                self.is_initial_phase = False
                
            # Determine batch size based on remaining budget and current evaluation count
            remaining_budget = self.budget - self.n_evals
            if remaining_budget < 5:
                batch_size = 1
            elif remaining_budget < 20:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, max(1, remaining_budget // 20))
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically (every 5 evaluations or when we have enough data)
            if self.n_evals % 5 == 0 and len(self.X) > 10:
                try:
                    self.model = self._fit_model(self.X, self.y)
                except Exception as e:
                    print(f"Model fitting failed: {e}")
            
            # Check if we've reached the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()

```
The algorithm EGOABLBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4485 with standard deviation 0.0178.

took 9.41 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

