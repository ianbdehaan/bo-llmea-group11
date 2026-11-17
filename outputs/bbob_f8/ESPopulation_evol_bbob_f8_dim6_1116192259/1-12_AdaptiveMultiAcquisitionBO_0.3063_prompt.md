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
AdaptiveTrustRegionBO - This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np

class AdaptiveTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10)  # Initial points - start with more points for higher dimensions

        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Trust region parameters
        self.tr_radius = 1.0  # Initial trust region radius
        self.tr_center = None  # Center of current trust region
        
        # Convergence tracking
        self.best_y_history = []
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 5

    def _sample_points(self, n_points):
        # Sample points using Latin hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Using Gaussian Process with ARD Matern kernel
        # We use a fixed length scale for efficiency and avoid overfitting
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,  # Limited restarts for computational efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, best_y):
        # Implement expected improvement acquisition function
        # EI = (mu(x) - y_best) * Phi((mu(x) - y_best) / sigma(x)) + sigma(x) * phi((mu(x) - y_best) / sigma(x))
        
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)  # Avoid numerical issues
        
        # Calculate EI
        improvement = mu - best_y
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # 1. Sample random points within trust region
        # 2. Use L-BFGS-B optimization on acquisition function
        # 3. Select best points from both approaches
        
        n_random = max(10, batch_size // 2)  # Sample some random points
        n_optimize = batch_size - n_random
        
        # Sample random points within trust region
        if self.tr_center is not None:
            # Create bounds within trust region
            local_bounds = []
            for i in range(self.dim):
                lower = max(self.bounds[0][i], self.tr_center[i] - self.tr_radius)
                upper = min(self.bounds[1][i], self.tr_center[i] + self.tr_radius)
                local_bounds.append((lower, upper))
            
            # Generate random samples within trust region
            random_points = np.zeros((n_random, self.dim))
            for i in range(n_random):
                for j in range(self.dim):
                    random_points[i, j] = self.rng.uniform(local_bounds[j][0], local_bounds[j][1])
        else:
            # If no trust region defined yet, use full bounds
            random_points = self._sample_points(n_random)
        
        # Optimize acquisition function using L-BFGS-B from multiple starting points
        if self.X is not None and len(self.X) > 0:
            best_y = np.min(self.y)
            model = self._fit_model(self.X, self.y)
            
            # Generate starting points for optimization
            n_starts = min(5, max(1, n_optimize // 2))
            if self.tr_center is not None:
                start_points = np.zeros((n_starts, self.dim))
                for i in range(n_starts):
                    for j in range(self.dim):
                        start_points[i, j] = self.rng.uniform(
                            max(self.bounds[0][j], self.tr_center[j] - self.tr_radius),
                            min(self.bounds[1][j], self.tr_center[j] + self.tr_radius)
                        )
            else:
                start_points = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_opt_point = None
            
            for start_point in start_points:
                def obj(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x, model, best_y)[0, 0]
                
                try:
                    res = minimize(obj, start_point, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
                    if -res.fun > best_acq_val:
                        best_acq_val = -res.fun
                        best_opt_point = res.x
                except:
                    continue
            
            if best_opt_point is not None:
                optimize_points = best_opt_point.reshape(1, -1)
            else:
                optimize_points = self._sample_points(1)
        else:
            optimize_points = self._sample_points(1)
        
        # Combine points
        if n_optimize > 0 and len(optimize_points) > 0:
            selected_points = np.vstack([random_points, optimize_points[:n_optimize]])
        else:
            selected_points = random_points
        
        # Limit to batch_size
        if len(selected_points) > batch_size:
            selected_points = selected_points[:batch_size]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_values = []
        for point in X:
            y_values.append(func(point))
        
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
        
        # Initialize with Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        
        # Set initial trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Check if we should restart or shrink trust region due to no improvement
            if len(self.best_y_history) > 0:
                current_best = np.min(self.y)
                self.best_y_history.append(current_best)
                
                if current_best >= self.best_y_history[-2] - 1e-4:  # No significant improvement
                    self.consecutive_no_improvement += 1
                    if self.consecutive_no_improvement >= self.max_no_improvement:
                        # Restart with new random points
                        self.consecutive_no_improvement = 0
                        # Reduce trust region radius
                        self.tr_radius = max(self.tr_radius * 0.5, 0.1)
                else:
                    self.consecutive_no_improvement = 0
            else:
                self.best_y_history.append(np.min(self.y))
            
            # Select next points to evaluate
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size to remaining budget
            if batch_size <= 0:
                break
                
            next_points = self._select_next_points(batch_size)
            
            # Evaluate new points
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            
            # Update trust region center to the best point so far
            best_idx = np.argmin(self.y)
            self.tr_center = self.X[best_idx].copy()
            
            # Possibly increase trust region radius if progress is good
            if len(self.best_y_history) > 1 and self.best_y_history[-1] < self.best_y_history[-2]:
                self.tr_radius = min(self.tr_radius * 1.1, 5.0)  # Increase trust region but cap it
            
            # Early termination for small budgets to avoid excessive overhead
            if self.n_evals >= self.budget - 1:
                break
                
        # Return the best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()

```
The algorithm AdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3392 with standard deviation 0.0129.

took 10.35 seconds to run.

Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

