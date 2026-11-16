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
- EMBO: 0.4272, 190.68 seconds, **Efficient Multi-Start Bayesian Optimization with Adaptive Exploration (EMBO)**

This is a novel Bayesian optimization algorithm that combines several key innovations to achieve excellent performance on black-box optimization problems. The approach uses an adaptive exploration strategy, multi-start acquisition function optimization, and efficient surrogate modeling.

Key characteristics:
1. **Adaptive Exploration**: Uses a dynamic exploration weight in the lower confidence bound (LCB) acquisition function that decreases over time
2. **Multi-Start Optimization**: Employs multiple random restarts when optimizing the acquisition function to avoid local optima
3. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
4. **Adaptive Initial Design**: Starts with a maximin Latin Hypercube design scaled to the problem dimensionality
5. **Batch Selection**: Selects multiple points in batches using diverse sampling strategies


- AdaptiveMultiSurrogateBO: 0.0000, 0.00 seconds, **AdaptiveMultiSurrogateBO**: This novel Bayesian Optimization algorithm combines multiple surrogate models with adaptive selection and a sophisticated acquisition function strategy. It uses an initial Latin Hypercube design, employs both Gaussian Process (with ARD Matern kernel) and Random Forest surrogates for robust modeling, implements a multi-acquisition function portfolio that dynamically switches between Expected Improvement, Lower Confidence Bound, and Thompson Sampling, and utilizes an efficient trust region-based acquisition optimizer with restarts to balance exploration and exploitation effectively.


- ATRBO: 0.0000, 0.00 seconds, **Adaptive Trust-Region Bayesian Optimization (ATRBO)**

This is a novel Bayesian optimization algorithm that combines adaptive trust-region strategies with multi-point acquisition functions. The key characteristics are:
1. Uses a trust-region mechanism that dynamically adjusts based on function landscape changes
2. Employs an ensemble of acquisition functions to balance exploration and exploitation
3. Implements efficient local search around promising regions
4. Uses a sophisticated initial design (Latin hypercube with optimization)
5. Adapts the surrogate model complexity based on evaluation progress


- AdaptiveBO: 0.0000, 0.00 seconds, Adaptive Bayesian Optimization with Multi-Start Acquisition Function Optimization and Dynamic Exploration

This is a novel Bayesian optimization algorithm that combines several key innovations:
1. Adaptive initial design using Latin Hypercube sampling with dynamic size based on dimensionality
2. Robust Gaussian Process surrogate model with automatic relevance detection (ARD) Matern kernel
3. Multi-start acquisition function optimization to avoid local optima
4. Dynamic exploration-exploitation balance through a schedule that reduces exploration over time
5. Efficient batch selection using diverse candidate sampling followed by local refinement
6. Adaptive budget allocation that prioritizes early exploration and later exploitation




The selected solution to update is:
**AdaptiveMultiSurrogateBO**: This novel Bayesian Optimization algorithm combines multiple surrogate models with adaptive selection and a sophisticated acquisition function strategy. It uses an initial Latin Hypercube design, employs both Gaussian Process (with ARD Matern kernel) and Random Forest surrogates for robust modeling, implements a multi-acquisition function portfolio that dynamically switches between Expected Improvement, Lower Confidence Bound, and Thompson Sampling, and utilizes an efficient trust region-based acquisition optimizer with restarts to balance exploration and exploitation effectively.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
import warnings

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
        self.n_init = max(2 * dim + 1, 10) # Initial design size
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Surrogate model parameters
        self.gp_model = None
        self.rf_model = None
        self.model_weights = [0.5, 0.5]  # GP and RF weights
        
        # Acquisition function parameters
        self.acq_func_type = "ei"  # ei, lcb, thompson
        self.lcb_weight = 0.1
        self.acq_func_history = []
        
        # Trust region parameters
        self.tr_radius = 1.0
        self.tr_center = None

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for initial design
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit both GP and RF models
        # Gaussian Process with ARD Matern kernel
        try:
            # Use a more robust initialization for GP
            kernel = Matern(length_scale=np.ones(self.dim), length_scale_bounds=[(0.1, 10)]*self.dim, nu=2.5)
            # Add WhiteKernel for noise modeling
            kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
            
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            self.gp_model.fit(X, y)
        except Exception as e:
            warnings.warn(f"GP fitting failed: {e}")
            self.gp_model = None
            
        # Random Forest
        try:
            self.rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=int(self.rng.integers(2**31 - 1)),
                n_jobs=1
            )
            self.rf_model.fit(X, y.ravel())
        except Exception as e:
            warnings.warn(f"RF fitting failed: {e}")
            self.rf_model = None

    def _acquisition_function(self, X):
        # Calculate acquisition function using ensemble of models
        if self.gp_model is not None:
            try:
                # Expected Improvement with GP model
                mean, std = self.gp_model.predict(X, return_std=True)
                best_y = np.min(self.y)
                improvement = best_y - mean
                # EI formula
                Z = improvement / (std + 1e-8)
                ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
                ei = np.clip(ei, -1e10, 1e10)
            except Exception as e:
                warnings.warn(f"GP EI calculation failed: {e}")
                ei = np.zeros((X.shape[0], 1))
        else:
            ei = np.zeros((X.shape[0], 1))
            
        if self.rf_model is not None:
            try:
                # Predict with RF model
                rf_pred = self.rf_model.predict(X)
                best_y = np.min(self.y)
                rf_improvement = best_y - rf_pred
                rf_ei = np.maximum(rf_improvement, 0).reshape(-1, 1)
            except Exception as e:
                warnings.warn(f"RF EI calculation failed: {e}")
                rf_ei = np.zeros((X.shape[0], 1))
        else:
            rf_ei = np.zeros((X.shape[0], 1))
            
        # Ensemble combination
        combined_ei = self.model_weights[0] * ei + self.model_weights[1] * rf_ei
        
        return combined_ei

    def _select_next_points(self, batch_size):
        # Select next points using trust region sampling with restarts
        if self.X is None or len(self.X) < 2:
            # For initial points, use random sampling within bounds
            return self._sample_points(batch_size)
        
        # Initialize best point for trust region center
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Use trust region based on current best point and adaptively adjusted radius
        points_to_evaluate = []
        
        # Sample multiple candidates around the trust region
        n_candidates = max(50, batch_size * 10)
        candidates = self._sample_points(n_candidates)
        
        # Filter candidates within trust region
        distances = np.linalg.norm(candidates - self.tr_center, axis=1)
        valid_indices = np.where(distances <= self.tr_radius)[0]
        valid_candidates = candidates[valid_indices]
        
        if len(valid_candidates) < batch_size:
            # Expand the trust region or sample more points
            valid_candidates = candidates[:batch_size]
            
        # Select the best candidates based on acquisition function
        acq_values = self._acquisition_function(valid_candidates)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        
        return valid_candidates[top_indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception as e:
                warnings.warn(f"Function evaluation failed for point {x}: {e}")
                # Return a large value to penalize bad points
                y_vals.append(1e10)
                
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
        
        # Initialize with Latin Hypercube design
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Fit surrogate models
        self._fit_model(self.X, self.y)
        
        # Iteratively optimize
        batch_size = 1  # Single point evaluation at a time
        
        while self.n_evals < self.budget - batch_size:
            if self.X is None or len(self.X) < 2:
                break
                
            # Fit models after each batch
            self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate points
            next_values = self._evaluate_points(func, next_points)
            
            # Update points
            self._update_eval_points(next_points, next_values)
            
            # Adjust trust region radius based on progress
            if len(self.y) >= 5:
                recent_improvement = np.min(self.y[-5:]) - np.min(self.y)
                if recent_improvement < 0.01 and self.tr_radius > 0.1:
                    self.tr_radius *= 0.8  # Shrink trust region
                elif recent_improvement > 0.1 and self.tr_radius < 2.0:
                    self.tr_radius *= 1.2  # Expand trust region
            
            # Adaptive acquisition function switching based on recent performance
            if len(self.acq_func_history) >= 5:
                recent_acq = np.array(self.acq_func_history[-5:])
                avg_improvement = np.mean(np.diff(recent_acq))
                if avg_improvement < -0.1:  # If not improving much, switch to LCB
                    self.acq_func_type = "lcb"
                elif avg_improvement > 0.1:
                    self.acq_func_type = "ei"  # Return to EI if improving
                else:
                    # Keep current type or randomly switch occasionally
                    if self.rng.random() < 0.1:
                        self.acq_func_type = "thompson" if self.acq_func_type == "ei" else "ei"
        
        # Final evaluation of best point
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

```
An error occurred : [Errno Evaluation timed out (%d)] 120

Replace the acquisition function optimizer of the selected solution by a substantially different alternative (options: L-BFGS-B from several starting points, differential evolution, a small evolution strategy, random candidates plus local refinement, candidates sampled around the best points), and adapt the rest of the algorithm where needed. Explain in the justification why the new acquisition function optimizer should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

