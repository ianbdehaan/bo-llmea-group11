You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on BBOB function 8 (Rosenbrock, original), instance 1, dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
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
- AdaptiveTrustRegionBO: 0.3392, 10.35 seconds, AdaptiveTrustRegionBO - This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- EfficientRobustBO: 0.3375, 24.86 seconds, **Efficient Robust Bayesian Optimization (ERBO)**: This algorithm combines a robust initialization strategy with an adaptive surrogate model and acquisition function to efficiently navigate high-dimensional black-box optimization. It uses a maximin Latin hypercube design for initial sampling, employs a sparse Gaussian process with ARD Matern kernel for modeling, and implements a multi-start L-BFGS optimizer for acquisition function maximization. The algorithm incorporates uncertainty-aware exploration through a novel acquisition function that balances global search with local exploitation.


- AdaptiveSurrogateBO: 0.2989, 2.98 seconds, AdaptiveSurrogateBO: This Bayesian Optimization algorithm employs an adaptive surrogate model selection strategy that dynamically switches between Gaussian Process (GP) and Random Forest (RF) models based on performance metrics. It uses a novel hybrid acquisition function combining Expected Improvement with a local exploration term, optimized via a multi-start L-BFGS approach with intelligent restarts. The algorithm implements an efficient initialization using a maximin Latin Hypercube design and employs a progressive refinement strategy that adapts the search space based on observed improvements.


- AdaptiveMultiSurrogateBO: 0.0000, 0.00 seconds, AdaptiveMultiSurrogateBO - This algorithm employs an adaptive Bayesian optimization approach that dynamically switches between multiple surrogate models and acquisition functions. It starts with a diverse initial design using Latin hypercube sampling, then adaptively selects between Expected Improvement (EI), Probability of Improvement (PI), and Lower Confidence Bound (LCB) acquisition functions based on performance metrics. The surrogate model uses a Gaussian Process with anisotropic Matern kernel that is retrained periodically with a hybrid approach combining old and new data. A trust region mechanism is employed to focus exploration near promising regions while maintaining global search capabilities.




The selected solution to update is:
AdaptiveSurrogateBO: This Bayesian Optimization algorithm employs an adaptive surrogate model selection strategy that dynamically switches between Gaussian Process (GP) and Random Forest (RF) models based on performance metrics. It uses a novel hybrid acquisition function combining Expected Improvement with a local exploration term, optimized via a multi-start L-BFGS approach with intelligent restarts. The algorithm implements an efficient initialization using a maximin Latin Hypercube design and employs a progressive refinement strategy that adapts the search space based on observed improvements.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from sklearn.preprocessing import StandardScaler
import warnings

class AdaptiveSurrogateBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * self.dim + 1, 10) # Initial points based on dimensionality
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model and acquisition function parameters
        self.model_type = 'gp'  # Start with GP
        self.model = None
        self.scaler = StandardScaler()
        self.best_y = np.inf
        self.best_x = None
        
    def _sample_points(self, n_points):
        # Use maximin Latin Hypercube design for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = qmc.scale(points, self.bounds[0], self.bounds[1])
        return points
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model based on performance
        # Use GP with Matern kernel for smooth functions like Rosenbrock
        if self.model_type == 'gp':
            try:
                # Try different kernels for robustness
                kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=2,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X, y)
                return gp
            except:
                # Fallback to RF if GP fails
                self.model_type = 'rf'
                rf = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=int(self.rng.integers(2**31 - 1)),
                    n_jobs=1
                )
                rf.fit(X, y)
                return rf
        else:
            # Random Forest for robustness
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=int(self.rng.integers(2**31 - 1)),
                n_jobs=1
            )
            rf.fit(X, y)
            return rf
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function: Expected Improvement + Local Exploration
        if self.model_type == 'gp':
            # Expected Improvement for GP
            try:
                # Get predictions from GP
                mean, std = self.model.predict(X, return_std=True)
                
                # Calculate EI with exploration factor
                improvement = self.best_y - mean - 1e-8
                ei = improvement * norm.cdf(improvement/std) + std * norm.pdf(improvement/std)
                
                # Add local exploration term
                # Local exploration via uncertainty-weighted term
                local_exploration = std * 0.5
                
                return ei.reshape(-1, 1) + local_exploration.reshape(-1, 1)
            except:
                # Fallback to simple EI if there's an issue
                return np.ones((X.shape[0], 1)) * 0.1
        else:
            # For RF, use variance as exploration measure
            try:
                predictions = np.array([self.model.predict(X) for _ in range(10)])
                variance = np.var(predictions, axis=0)
                # Simple acquisition based on variance (higher variance = more exploration)
                return variance.reshape(-1, 1)
            except:
                return np.ones((X.shape[0], 1)) * 0.1
    
    def _select_next_points(self, batch_size):
        # Multi-start L-BFGS optimization of acquisition function
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple random starting points for optimization
        n_starts = min(20, max(5, self.n_evals // 2))
        
        for _ in range(n_starts):
            # Start from a random point within bounds
            start_point = self._sample_points(1)[0]
            
            def acq_func(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0][0]
            
            try:
                # Use L-BFGS-B optimization with bounds
                res = minimize(
                    acq_func, 
                    start_point,
                    method='L-BFGS-B',
                    bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                    options={'maxiter': 50}
                )
                
                if res.success:
                    val = -res.fun
                    if val > best_acq_val:
                        best_acq_val = val
                        best_x = res.x.reshape(1, -1)
            except:
                continue
        
        # If no good points found, return a random point
        if best_x is None:
            return self._sample_points(batch_size)
        
        return np.repeat(best_x, batch_size, axis=0)
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None:
            return np.array([])
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except:
                # If evaluation fails, use a large value
                y_vals.append(np.inf)
        
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
        
        # Update best point found so far
        min_idx = np.argmin(self.y)
        if self.y[min_idx] < self.best_y:
            self.best_y = self.y[min_idx]
            self.best_x = self.X[min_idx]
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube design
        init_X = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_X)
        self._update_eval_points(init_X, init_y)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y.flatten())
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, reduce batch size
            remaining_budget = self.budget - self.n_evals
            batch_size = min(10, remaining_budget)
            
            if batch_size <= 0:
                break
                
            # Select next points to evaluate
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            
            # Update the model with new data
            self._update_eval_points(next_X, next_y)
            
            # Re-fit the model periodically (every 5 evaluations) to maintain accuracy
            if self.n_evals % 5 == 0:
                try:
                    self.model = self._fit_model(self.X, self.y.flatten())
                except:
                    pass  # If fitting fails, keep current model
        
        return self.best_y, self.best_x

```
The algorithm AdaptiveSurrogateBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2989 with standard deviation 0.0185.

took 2.98 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

