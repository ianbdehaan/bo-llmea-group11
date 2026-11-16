You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on the real-world problem 1124 (parameter estimation of Frequency Modulated Sound Waves), dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.
A BO algorithm can be built from many different components, for example:
- initial design: Sobol, Latin hypercube, random, maximin designs, different sizes (e.g. d+1 up to 4d points)
- surrogate model: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y, random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted only near the best point
- acquisition function: expected improvement, log expected improvement, lower confidence bound with a schedule for the exploration weight, probability of improvement, Thompson sampling from the GP posterior, a portfolio that switches between several acquisition functions
- acquisition function optimizer: L-BFGS-B from several starting points, differential evolution, a small evolution strategy, random candidates plus local refinement, candidates sampled around the best points
- exploration strategy: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points
Do not always use the same default choices (e.g. a scikit-learn GP with expected improvement on random candidates); choose the components that suit the problem.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EMBO_TRE: 0.4619, 532.41 seconds, **Efficient Multi-Start Bayesian Optimization with Trust-Region Exploration (EMBO-TRE)**

This is a novel Bayesian optimization algorithm that combines several key innovations to achieve excellent performance on black-box optimization problems. The approach uses an adaptive trust-region exploration strategy, multi-start acquisition function optimization, and efficient surrogate modeling.

Key characteristics:
1. **Trust-Region Exploration**: Uses a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
2. **Multi-Start Optimization**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima  
3. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
4. **Adaptive Initial Design**: Starts with a maximin Latin Hypercube design scaled to the problem dimensionality
5. **Batch Selection**: Selects multiple points in batches using diverse sampling strategies within trust regions


- EMBO_SB_Enhanced: 0.4561, 209.45 seconds, **Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB) - Enhanced Version**

This refined Bayesian optimization algorithm builds upon the previous EMBO-SB framework but incorporates several key improvements to achieve better performance. The main enhancements include:
1. **Enhanced Initial Design**: Uses a maximin Latin Hypercube design with adaptive scaling based on problem dimensionality and budget
2. **Improved Surrogate Model**: Implements a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel optimized with efficient hyperparameter restarts
3. **Adaptive Acquisition Function**: Combines Expected Improvement with Lower Confidence Bound using dynamic weights that adapt based on progress through the budget
4. **Smart Batch Selection Strategy**: Implements an adaptive batch sizing mechanism that decreases as budget is exhausted, and uses diverse sampling strategies for batch points
5. **Robust Acquisition Optimization**: Employs a multi-start L-BFGS optimizer with diverse starting points including trust-region boundaries and recent best points
6. **Progress-Driven Exploration Scheduling**: Uses a dynamic exploration weight schedule that adapts based on optimization progress rather than fixed time intervals


- EMBO_SB: 0.4558, 53.85 seconds, **Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB)**

This refined Bayesian optimization algorithm improves upon the original EMBO by incorporating smart batch selection strategies, more efficient acquisition function optimization, and better handling of early iterations. Key improvements include:
1. Adaptive batch sizing that decreases as the budget is exhausted
2. Improved initial design using maximin Latin Hypercube with dimension scaling
3. Enhanced acquisition function optimizer using multi-start L-BFGS with diverse starting points
4. Dynamic exploration weight scheduling based on progress rather than fixed time intervals
5. Efficient surrogate model updates with smart hyperparameter optimization restarts
6. Better handling of early iterations with more robust initial exploration


- EMBO_SB: 0.4555, 143.60 seconds, **Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB)**

This refined Bayesian optimization algorithm improves upon the original EMBO by incorporating smart batch selection strategies, more efficient acquisition function optimization, and better handling of early iterations. Key improvements include:
1. Adaptive batch sizing that decreases as the budget is exhausted
2. Improved initial design using maximin Latin Hypercube with dimension scaling
3. Enhanced acquisition function optimizer using multi-start L-BFGS with diverse starting points
4. Dynamic exploration weight scheduling based on progress rather than fixed time intervals
5. Efficient surrogate model updates with smart hyperparameter optimization restarts
6. Better handling of early iterations with more robust initial exploration




The selected solution to update is:
**Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB)**

This refined Bayesian optimization algorithm improves upon the original EMBO by incorporating smart batch selection strategies, more efficient acquisition function optimization, and better handling of early iterations. Key improvements include:
1. Adaptive batch sizing that decreases as the budget is exhausted
2. Improved initial design using maximin Latin Hypercube with dimension scaling
3. Enhanced acquisition function optimizer using multi-start L-BFGS with diverse starting points
4. Dynamic exploration weight scheduling based on progress rather than fixed time intervals
5. Efficient surrogate model updates with smart hyperparameter optimization restarts
6. Better handling of early iterations with more robust initial exploration


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class EMBO_SB:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10) # Adaptive initial points

        # Random generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model parameters
        self.gp = None
        
        # Acquisition function parameters
        # Dynamic exploration weight based on progress through budget
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use a small number of restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Lower Confidence Bound with adaptive exploration weight
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current exploration weight (adaptive)
        if len(self.acq_weight_schedule) > 0:
            # Use progress-based weight schedule
            progress = min(1.0, self.n_evals / max(1, self.budget * 0.7))
            # Map progress to appropriate weight from schedule
            weight_idx = int(progress * (len(self.acq_weight_schedule) - 1))
            exploration_weight = self.acq_weight_schedule[min(weight_idx, len(self.acq_weight_schedule) - 1)]
        else:
            exploration_weight = 0.5
            
        # LCB acquisition function
        lcb = mean - exploration_weight * std
        
        return lcb.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using multiple starting points with L-BFGS"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            try:
                # Use L-BFGS with bounds for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0=start_point[0],
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization"""
        # For small batches, use a more focused approach
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use multi-start L-BFGS to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=10)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using random sampling with some structure
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample from a region around the current best point to maintain diversity
                if self.X is not None and len(self.X) > 0:
                    # Use recent best points to define search region
                    recent_best_idx = np.argsort(self.y.flatten())[:min(5, len(self.y))]
                    recent_best_points = self.X[recent_best_idx]
                    # Add some variation by sampling from a local region around the best point
                    if len(recent_best_points) > 0:
                        center = recent_best_points[np.argmin(self.y[recent_best_idx])]
                        # Sample around the center point with some randomness
                        for _ in range(remaining):
                            sample_point = center + self.rng.normal(0, 0.2, self.dim)
                            # Clip to bounds
                            sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                            selected_points.append(sample_point.reshape(1, -1))
                    else:
                        # Fallback to random sampling
                        for _ in range(remaining):
                            selected_points.append(self._sample_points(1))
                else:
                    # Fallback to random sampling
                    for _ in range(remaining):
                        selected_points.append(self._sample_points(1))
            
            return np.vstack(selected_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
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
        
        # Initialize with maximin Latin Hypercube design scaled to problem dimensionality
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

```
The algorithm EMBO_SB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4558 with standard deviation 0.0192.

took 53.85 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

