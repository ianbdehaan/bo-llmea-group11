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
AdaptiveMultiSurrogateBO - This algorithm employs an adaptive Bayesian optimization approach that dynamically switches between multiple surrogate models and acquisition functions. It starts with a diverse initial design using Latin hypercube sampling, then adaptively selects between Expected Improvement (EI), Probability of Improvement (PI), and Lower Confidence Bound (LCB) acquisition functions based on performance metrics. The surrogate model uses a Gaussian Process with anisotropic Matern kernel that is retrained periodically with a hybrid approach combining old and new data. A trust region mechanism is employed to focus exploration near promising regions while maintaining global search capabilities.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler

class AdaptiveMultiSurrogateBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * self.dim + 1, 10)  # Initial points based on dimensionality
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model and acquisition function tracking
        self.model = None
        self.acq_func_history = []
        self.acq_func_weights = [0.33, 0.33, 0.34]  # EI, PI, LCB weights
        self.acq_func_names = ['ei', 'pi', 'lcb']
        self.trust_region_center = None
        self.trust_region_radius = 2.0
        
    def _sample_points(self, n_points):
        # Use Latin hypercube sampling for diverse initial points
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use Gaussian Process with anisotropic Matern kernel
        # Add noise to avoid numerical issues
        
        # Scale the data for better GP performance
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Define kernel with automatic relevance determination (ARD)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Fit GP model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise for numerical stability
            n_restarts_optimizer=3,  # Few restarts to save computation
            normalize_y=False,  # We already normalized
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_scaled, y_scaled)
        
        # Store scalers and model for later use
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model = gp
        
        return gp
    
    def _acquisition_function(self, X, acq_func_name):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        
        if self.model is None:
            # If no model, return random values
            return self.rng.random(len(X))
        
        # Transform X to scaled space
        X_scaled = self.scaler_X.transform(X)
        
        # Get predictions from GP
        mean, std = self.model.predict(X_scaled, return_std=True)
        
        # Get best observed value (from original scale)
        y_best = np.min(self.y) if len(self.y) > 0 else 0.0
        
        # Transform y_best to scaled space
        y_best_scaled = self.scaler_y.transform([[y_best]]).ravel()[0]
        
        if acq_func_name == 'ei':
            # Expected Improvement
            improvement = y_best_scaled - mean
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return ei
            
        elif acq_func_name == 'pi':
            # Probability of Improvement
            improvement = y_best_scaled - mean
            z = improvement / (std + 1e-8)
            pi = norm.cdf(z)
            return pi
            
        elif acq_func_name == 'lcb':
            # Lower Confidence Bound with exploration weight
            kappa = 2.0  # Exploration parameter
            lcb = mean - kappa * std
            return lcb
        
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func_name}")
    
    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        candidates = []
        
        # Strategy 1: Sample from trust region if available
        if self.trust_region_center is not None:
            # Generate samples around trust region center
            region_samples = self._sample_points(20)
            # Filter by trust region
            distances = np.linalg.norm(region_samples - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            if len(valid_indices) > 0:
                candidates.extend(region_samples[valid_indices[:10]])
        
        # Strategy 2: Sample from full space for diversity
        full_space_samples = self._sample_points(30)
        candidates.extend(full_space_samples)
        
        # Strategy 3: Use local optimization around best points
        if len(self.X) > 0:
            # Sample from the neighborhood of best points
            best_indices = np.argsort(self.y)[:min(5, len(self.y))]
            for i in best_indices:
                # Local search around best points
                point = self.X[i]
                local_samples = self._sample_points(3)
                # Add small perturbations to original point
                perturbed = point + (self.rng.random(point.shape) - 0.5) * 0.2
                candidates.append(perturbed)
        
        # Ensure we have enough candidates
        if len(candidates) < 10:
            candidates.extend(self._sample_points(10 - len(candidates)))
            
        candidates = np.array(candidates[:50])  # Limit candidates
        
        # Evaluate acquisition function for all candidates using the selected acquisition function
        acq_values = []
        for func_name in self.acq_func_names:
            values = self._acquisition_function(candidates, func_name)
            acq_values.append(values)
        
        # Weighted combination of acquisition functions
        combined_acq = np.zeros(len(candidates))
        for i, (func_name, weight) in enumerate(zip(self.acq_func_names, self.acq_func_weights)):
            combined_acq += weight * acq_values[i]
        
        # Select top candidates based on combined acquisition values
        selected_indices = np.argsort(combined_acq)[-batch_size:]
        next_points = candidates[selected_indices]
        
        return next_points
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        # Evaluate all points at once to maximize efficiency
        y_values = []
        for point in X:
            y_val = func(point)
            y_values.append(y_val)
        
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
            
        # Update trust region center based on best point
        if len(self.y) > 0:
            best_idx = np.argmin(self.y)
            self.trust_region_center = self.X[best_idx]
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin hypercube sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to proceed
            if self.n_evals >= self.budget:
                break
                
            # Select next points to evaluate (batch size = 1 for simplicity)
            batch_size = min(1, self.budget - self.n_evals)
            X_next = self._select_next_points(batch_size)
            
            # Evaluate next points
            y_next = self._evaluate_points(func, X_next)
            
            # Update model with new points
            self._update_eval_points(X_next, y_next)
            
            # Retrain the model periodically or when we have enough data
            if len(self.X) >= self.n_init + 5:
                self._fit_model(self.X, self.y)
                
        # Return best result
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()

```
An error occurred : Traceback (most recent call last):
  File "<AdaptiveMultiSurrogateBO>", line 221, in __call__
 221->             X_next = self._select_next_points(batch_size)
  File "<AdaptiveMultiSurrogateBO>", line 152, in _select_next_points
 150 |             candidates.extend(self._sample_points(10 - len(candidates)))
 151 |             
 152->         candidates = np.array(candidates[:50])  # Limit candidates
 153 |         
 154 |         # Evaluate acquisition function for all candidates using the selected acquisition function
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (35,) + inhomogeneous part.


Replace the exploration strategy of the selected solution by a substantially different alternative (options: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points), and adapt the rest of the algorithm where needed. Explain in the justification why the new exploration strategy should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

