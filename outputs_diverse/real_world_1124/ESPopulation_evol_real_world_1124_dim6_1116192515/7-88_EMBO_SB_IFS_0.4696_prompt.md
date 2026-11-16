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
- EMBO_SB: 0.4699, 88.72 seconds, **Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Random Forest Surrogate (EMBO-SB)**

This novel Bayesian optimization algorithm replaces the Latin Hypercube initial design with a Sobol sequence for superior space-filling properties, while maintaining the efficient Random Forest surrogate model and multi-start acquisition function optimization. The approach uses a Sobol sequence for the initial design points, which provides better uniformity across the search space compared to Latin Hypercube sampling, especially in higher dimensions.


- EMBO_SB_TS: 0.4684, 281.21 seconds, **Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Thompson Sampling (EMBO-SB-TS)**

This novel Bayesian optimization algorithm combines several key innovations to achieve excellent performance on black-box optimization problems:

1. **Sobol Sequence Initial Design**: Replaces Latin Hypercube sampling with Sobol sequences for superior space-filling properties, especially in higher dimensions
2. **Thompson Sampling Acquisition Function**: Uses Thompson sampling from the GP posterior for exploration-exploitation balance instead of expected improvement
3. **Multi-Start Optimization with Diverse Starting Points**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima  
4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
5. **Adaptive Trust-Region Exploration**: Uses a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
6. **Batch Selection with Diversity**: Selects multiple points in batches using diverse sampling strategies within trust regions


- EMBO_SB_TS: 0.4684, 278.00 seconds, **Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Thompson Sampling (EMBO-SB-TS)**

This novel Bayesian optimization algorithm combines several key innovations to achieve excellent performance on black-box optimization problems:

1. **Sobol Sequence Initial Design**: Replaces the Latin Hypercube sampling with Sobol sequences for superior space-filling properties, especially in higher dimensions
2. **Thompson Sampling Acquisition Function**: Uses Thompson sampling from the GP posterior for exploration-exploitation balance instead of expected improvement
3. **Multi-Start Optimization with Diverse Starting Points**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima
4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
5. **Adaptive Trust-Region Exploration**: Implements a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
6. **Batch Selection with Diversity**: Selects multiple points in batches using diverse sampling strategies within trust regions


- EMBO_SOBOL_TRE_TS: 0.4684, 278.05 seconds, **Efficient Multi-Start Bayesian Optimization with Sobol Initial Design, Trust-Region Exploration, and Thompson Sampling (EMBO-SOBOL-TRE-TS)**

This is a novel Bayesian optimization algorithm that combines several key innovations to achieve excellent performance on black-box optimization problems. The approach improves upon EMBO_TRE_TS by replacing the Latin Hypercube initial design with a Sobol sequence for superior space-filling properties, while maintaining the efficient Thompson sampling acquisition function and adaptive trust-region exploration strategy.

Key characteristics:
1. **Sobol Sequence Initial Design**: Uses Sobol sequences for the initial design points, providing better uniformity across the search space compared to Latin Hypercube sampling
2. **Trust-Region Exploration**: Implements a dynamic trust-region mechanism around the best point that grows and shrinks based on progress  
3. **Multi-Start Optimization**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima
4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
5. **Thompson Sampling Acquisition Function**: Uses Thompson sampling from the GP posterior for exploration-exploitation balance
6. **Adaptive Batch Selection**: Selects multiple points in batches using diverse sampling strategies within trust regions




The selected solution to update is:
**Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Random Forest Surrogate (EMBO-SB)**

This novel Bayesian optimization algorithm replaces the Latin Hypercube initial design with a Sobol sequence for superior space-filling properties, while maintaining the efficient Random Forest surrogate model and multi-start acquisition function optimization. The approach uses a Sobol sequence for the initial design points, which provides better uniformity across the search space compared to Latin Hypercube sampling, especially in higher dimensions.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
        
        # RF model parameters
        self.rf = None
        
        # Acquisition function parameters
        # Dynamic exploration weight based on progress through budget
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Random Forest regressor
        # Use a moderate number of trees for efficiency
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # No parallel processing to maintain reproducibility
        )
        
        rf.fit(X, y.flatten())
        return rf

    def _acquisition_function(self, X):
        # Lower Confidence Bound with adaptive exploration weight using RF predictions
        if self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from RF (mean and std)
        mean = self.rf.predict(X)
        
        # Get prediction variances (uncertainty) for acquisition function
        # For Random Forest, we can estimate uncertainty through the variance of predictions
        # We'll use a small constant as uncertainty for now
        
        # Alternative approach: Estimate uncertainty from the training data spread
        if self.y is not None:
            # Use the standard deviation of training targets as an estimate of noise level
            std = np.std(self.y) if len(self.y) > 0 else 1.0
        else:
            std = 1.0
            
        # Since Random Forest doesn't provide natural uncertainty estimates,
        # we'll use a heuristic based on the range of function values seen so far
        if self.y is not None and len(self.y) > 1:
            # Use the range as a proxy for uncertainty (simplified approach)
            uncertainty = np.max(self.y) - np.min(self.y) if np.max(self.y) != np.min(self.y) else 1.0
        else:
            uncertainty = 1.0
            
        # For better uncertainty estimation, we could use bootstrap sampling or 
        # implement a custom RF with variance estimates, but for efficiency we'll use this approach
        
        # Use a fixed uncertainty multiplier to control exploration
        uncertainty_multiplier = 0.5  # Adjust as needed
        
        # LCB acquisition function using mean and uncertainty estimate
        lcb = mean - uncertainty_multiplier * uncertainty
        
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
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.rf = self._fit_model(self.X, self.y)
        
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
            self.rf = self._fit_model(self.X, self.y)
            
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
The algorithm EMBO_SB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4699 with standard deviation 0.0152.

took 88.72 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

