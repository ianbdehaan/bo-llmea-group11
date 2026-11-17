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
**Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Thompson Sampling (EMBO-SB-TS)**

This novel Bayesian optimization algorithm combines several key innovations to achieve excellent performance on black-box optimization problems:

1. **Sobol Sequence Initial Design**: Replaces the Latin Hypercube sampling with Sobol sequences for superior space-filling properties, especially in higher dimensions
2. **Thompson Sampling Acquisition Function**: Uses Thompson sampling from the GP posterior for exploration-exploitation balance instead of expected improvement
3. **Multi-Start Optimization with Diverse Starting Points**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima
4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
5. **Adaptive Trust-Region Exploration**: Implements a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
6. **Batch Selection with Diversity**: Selects multiple points in batches using diverse sampling strategies within trust regions


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EMBO_SB_TS:
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
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.region_shrink_factor = 0.8
        self.region_expand_factor = 1.2
        self.progress_threshold = 1e-3
        self.last_improvement_evals = 0
        self.improvement_count = 0
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
    def _sample_points(self, n_points):
        # Use Sobol sequence for superior space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use fewer restarts for efficiency as recommended
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,  # Reduced from 3 to save computation
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Thompson Sampling acquisition function
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Sample from GP posterior
        samples = self.gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        return samples.reshape(-1, 1)

    def _get_trust_region_center(self):
        # Return the best point seen so far as the center of trust region
        if self.X is None or len(self.X) == 0:
            return np.mean(self.bounds, axis=0)
        else:
            best_idx = np.argmin(self.y)
            return self.X[best_idx].copy()

    def _update_trust_region(self):
        # Update trust region based on recent progress
        if len(self.performance_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        recent_improvement = (self.performance_history[-1] - self.performance_history[-2]) / abs(self.performance_history[-1])
        
        if recent_improvement > self.progress_threshold:
            # We're making good progress, expand the trust region
            self.trust_region_radius = min(
                self.max_trust_region_radius,
                self.trust_region_radius * self.region_expand_factor
            )
            self.improvement_count += 1
        else:
            # No significant improvement, shrink the trust region
            self.trust_region_radius = max(
                self.min_trust_region_radius,
                self.trust_region_radius * self.region_shrink_factor
            )
            self.improvement_count = 0

    def _sample_in_trust_region(self, center, n_points):
        # Sample points within trust region around the center
        # Create a box around center with size based on current trust region radius
        lower_bound = np.maximum(self.bounds[0], center - self.trust_region_radius)
        upper_bound = np.minimum(self.bounds[1], center + self.trust_region_radius)
        
        # If region is too small, just sample from bounds
        if np.all(upper_bound - lower_bound > 1e-6):
            # Use Sobol sampling within the trust region for better space-filling
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            points = sampler.random(n=n_points)
            points = lower_bound + points * (upper_bound - lower_bound)
        else:
            # Fall back to random sampling when trust region is very small
            points = self._sample_points(n_points)
            
        return points

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(15, max(3, batch_size * 2))  # Reduced starts to save computation
        
        # Get current trust region center and update it
        tr_center = self._get_trust_region_center()
        
        # Generate diverse starting points within trust region using Sobol for better distribution
        start_points = self._sample_in_trust_region(tr_center, n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=10,  # Reduced iterations for speed
                    popsize=3,   # Reduced population size
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:  # Minimize TS
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from trust region
            selected_points = self._sample_in_trust_region(tr_center, 1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points within trust region
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 3)):  # Reduced to save computation
                rand_point = self._sample_in_trust_region(tr_center, 1)
                candidates.append(rand_point)
                
        return np.vstack(candidates)[:batch_size]

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
        
        # Initialize with Sobol sequence design for better space-filling
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update trust region based on recent performance
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
                
                # Update trust region every few iterations or when we have enough data
                if (self.n_evals - self.last_improvement_evals > max(5, self.budget // 30)) and len(self.performance_history) >= 2:
                    self._update_trust_region()
                    self.last_improvement_evals = self.n_evals
            
            # Select next points to evaluate using acquisition function
            batch_size = min(3, self.budget - self.n_evals)  # Limit batch size for efficiency
            
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
The algorithm EMBO_SB_TS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4684 with standard deviation 0.0133.

took 278.00 seconds to run.

Replace the initial design of the selected solution by a substantially different alternative (options: Sobol, Latin hypercube, random, maximin designs, different sizes (e.g. d+1 up to 4d points)), and adapt the rest of the algorithm where needed. Explain in the justification why the new initial design should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

