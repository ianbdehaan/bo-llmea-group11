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


- EMBO_SB: 0.4558, 53.85 seconds, **Efficient Multi-Start Bayesian Optimization with Adaptive Exploration and Smart Batch Selection (EMBO-SB)**

This refined Bayesian optimization algorithm improves upon the original EMBO by incorporating smart batch selection strategies, more efficient acquisition function optimization, and better handling of early iterations. Key improvements include:
1. Adaptive batch sizing that decreases as the budget is exhausted
2. Improved initial design using maximin Latin Hypercube with dimension scaling
3. Enhanced acquisition function optimizer using multi-start L-BFGS with diverse starting points
4. Dynamic exploration weight scheduling based on progress rather than fixed time intervals
5. Efficient surrogate model updates with smart hyperparameter optimization restarts
6. Better handling of early iterations with more robust initial exploration


- ATRBORestart: 0.4528, 4.80 seconds, **Adaptive Trust-Region Bayesian Optimization with Restart Strategy (ATRBO-RS)**

This novel Bayesian optimization algorithm combines adaptive trust-region strategies with a restart mechanism when progress stagnates. Key characteristics include:
1. Uses Sobol sequence for initial sampling to ensure good space-filling properties across all dimensions
2. Implements an adaptive trust-region mechanism that dynamically adjusts based on function landscape changes
3. Incorporates a restart strategy that resets the trust region when optimization progress stalls
4. Employs an ensemble of acquisition functions (Expected Improvement, Lower Confidence Bound, Thompson Sampling) to balance exploration and exploitation
5. Uses a sparse Gaussian Process surrogate model with automatic relevance detection (ARD) Matern kernel for robust modeling
6. Adapts surrogate model complexity based on evaluation progress and uses efficient hyperparameter optimization


- ATRBOSobol: 0.4512, 4.55 seconds, **Adaptive Trust-Region Bayesian Optimization with Sobol Initial Design (ATRBO-Sobol)**

This is a novel Bayesian optimization algorithm that combines adaptive trust-region strategies with multi-point acquisition functions and a Sobol sequence-based initial design. The key characteristics are:
1. Uses Sobol sequence for initial sampling to ensure good space-filling properties across all dimensions
2. Employs an ensemble of acquisition functions (Expected Improvement, Lower Confidence Bound, Thompson Sampling) to balance exploration and exploitation
3. Implements efficient local search around promising regions with adaptive trust region adjustment
4. Uses a sparse Gaussian Process surrogate model with automatic relevance detection (ARD) Matern kernel for robust modeling
5. Adapts the surrogate model complexity based on evaluation progress




The selected solution to update is:
**Adaptive Trust-Region Bayesian Optimization with Restart Strategy (ATRBO-RS)**

This novel Bayesian optimization algorithm combines adaptive trust-region strategies with a restart mechanism when progress stagnates. Key characteristics include:
1. Uses Sobol sequence for initial sampling to ensure good space-filling properties across all dimensions
2. Implements an adaptive trust-region mechanism that dynamically adjusts based on function landscape changes
3. Incorporates a restart strategy that resets the trust region when optimization progress stalls
4. Employs an ensemble of acquisition functions (Expected Improvement, Lower Confidence Bound, Thompson Sampling) to balance exploration and exploitation
5. Uses a sparse Gaussian Process surrogate model with automatic relevance detection (ARD) Matern kernel for robust modeling
6. Adapts surrogate model complexity based on evaluation progress and uses efficient hyperparameter optimization


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
import warnings

class ATRBORestart:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.progress_window = 5  # Number of recent evaluations to check for progress
        self.min_progress_improvement = 0.01  # Minimum improvement to consider progress
        self.restart_threshold = 3  # Number of consecutive non-improving steps before restart
        self.restart_count = 0
        self.consecutive_no_improvement = 0
        self.min_trust_region = 0.05
        self.max_trust_region = 2.0
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling

    def _sample_points(self, n_points):
        # Use Sobol sequence sampling for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, seed=self.rng, scramble=True)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a sparse Gaussian Process with ARD Matern kernel
        # Use a small subset of data for training to keep computational cost low
        n_train = min(50, max(20, int(0.5 * len(X))))
        
        # If we have enough points, use a subset for fitting (sparse GP)
        if len(X) > 30:
            # Select a random subset of points for training to keep it fast
            train_indices = self.rng.choice(len(X), size=n_train, replace=False)
            X_train = X[train_indices]
            y_train = y[train_indices]
        else:
            X_train = X
            y_train = y
            
        # Use ARD Matern kernel with automatic relevance detection
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # Fit GP with noise estimation and optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_train, y_train)
        return gp

    def _acquisition_function(self, X):
        # Use an ensemble of acquisition functions weighted by their performance
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
        # Expected Improvement (EI)
        best_y = np.min(self.y)
        ei_values = np.zeros_like(mu)
        
        # Handle case where sigma is zero (no uncertainty)
        mask = sigma > 0
        if np.any(mask):
            improvement = best_y - mu[mask]
            z = improvement / sigma[mask]
            ei_values[mask] = improvement * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        # Lower Confidence Bound (LCB)
        kappa = 2.0  # Exploration parameter
        lcb = mu - kappa * sigma
        
        # Thompson sampling: sample from posterior distribution
        samples = gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        thompson = -samples.flatten()  # Negative because we minimize
        
        # Combine acquisition functions with weights
        acq_values = (
            self.acq_func_weights[0] * ei_values.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _select_next_points(self, batch_size):
        # Select points using a hybrid approach:
        # 1. First try to find the best point with acquisition function
        # 2. Then use local search around promising regions
        
        # Sample candidates for optimization
        n_candidates = min(1000, max(50, self.budget - self.n_evals))
        
        # Get points from current trust region if we have a center
        if self.trust_region_center is not None:
            # Generate points around the trust region center
            candidates = self._sample_points(n_candidates)
            
            # Apply trust region constraint: only consider points within radius
            distances = np.linalg.norm(candidates - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            if len(valid_indices) > 0:
                candidates = candidates[valid_indices]
            
            # If not enough candidates in trust region, expand the search
            if len(candidates) < batch_size:
                additional_candidates = self._sample_points(batch_size - len(candidates))
                candidates = np.vstack([candidates, additional_candidates])
        else:
            # First few iterations: random sampling with trust region expansion
            candidates = self._sample_points(n_candidates)
        
        # Limit candidates to reasonable size
        if len(candidates) > n_candidates:
            candidates = candidates[:n_candidates]
            
        # Evaluate acquisition function on candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points using top-k selection
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if len(X) == 0:
            return np.array([])
            
        y = []
        for point in X:
            try:
                val = func(point)
                y.append(val)
            except Exception as e:
                # Handle function evaluation errors by returning large value
                y.append(1e10)
                
        y = np.array(y).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            
        # Update trust region based on progress monitoring
        if len(self.y) > self.progress_window:
            # Check recent improvement
            recent_values = self.y[-self.progress_window:]
            old_values = self.y[:-self.progress_window]
            
            if len(old_values) > 0:
                best_old = np.min(old_values)
                best_recent = np.min(recent_values)
                
                # If no significant improvement, increment consecutive no improvement counter
                if (best_recent >= best_old - self.min_progress_improvement):
                    self.consecutive_no_improvement += 1
                else:
                    self.consecutive_no_improvement = 0
                    
                # If we've had enough non-improving steps, trigger restart
                if self.consecutive_no_improvement >= self.restart_threshold:
                    # Reset trust region to a wider scope
                    self.trust_region_radius = min(self.max_trust_region, self.trust_region_radius * 2.0)
                    self.consecutive_no_improvement = 0
                    self.restart_count += 1
                    
                    # Set center to a random point in the search space for global exploration
                    self.trust_region_center = self._sample_points(1)[0]
                    
            else:
                # If we don't have enough history, just reset if needed
                if self.consecutive_no_improvement >= self.restart_threshold:
                    self.trust_region_radius = min(self.max_trust_region, self.trust_region_radius * 2.0)
                    self.consecutive_no_improvement = 0
                    self.restart_count += 1
                    self.trust_region_center = self._sample_points(1)[0]
                
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Set trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.trust_region_center = self.X[best_idx]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, evaluate more points in parallel
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate these points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Update trust region center based on new best point
            if len(self.y) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx]
                
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()

```
The algorithm ATRBORestart got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4528 with standard deviation 0.0275.

took 4.80 seconds to run.

Use the selected solution only as a reference: design a new BO algorithm whose main idea is clearly different from all algorithms in the current population (for example a different surrogate model, acquisition function or search strategy), and explain the difference in the justification.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

