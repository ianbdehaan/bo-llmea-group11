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
- AdaptiveTrustRegionBO: 0.3687, 31.77 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- AdaptiveTrustRegionBO: 0.3672, 38.28 seconds, **AdaptiveTrustRegionBO**: This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- EfficientRobustBOMO: 0.3506, 44.08 seconds, **Efficient Robust Bayesian Optimization with Multi-Objective Acquisition Function (ERBO-MO)**: This novel Bayesian Optimization algorithm introduces a multi-objective framework for acquisition function optimization, combining expected improvement with local exploration and uncertainty-aware selection. It employs a sparse Gaussian process surrogate model with adaptive noise handling, uses a hybrid acquisition function that balances global search with local exploitation through multiple objectives, and optimizes the acquisition function using a diverse sampling strategy with local refinement. The algorithm incorporates dynamic adaptation of exploration vs exploitation based on progress through the budget and uses a robust initialization scheme.


- EfficientRobustBOAO: 0.3500, 22.52 seconds, **Efficient Robust Bayesian Optimization with Adaptive Multi-Objective Acquisition Function (ERBO-AO)**: This Bayesian Optimization algorithm introduces an adaptive multi-objective framework for acquisition function optimization, dynamically balancing expected improvement with uncertainty-aware selection. It employs a sparse Gaussian process surrogate model with adaptive noise handling, uses an adaptive hybrid acquisition function that adjusts exploration vs exploitation based on budget progress, and optimizes the acquisition function using a diverse sampling strategy with local refinement. The algorithm incorporates dynamic adaptation of exploration vs exploitation through a multi-objective optimization approach that evolves throughout the optimization process.




The selected solution to update is:
**Efficient Robust Bayesian Optimization with Adaptive Multi-Objective Acquisition Function (ERBO-AO)**: This Bayesian Optimization algorithm introduces an adaptive multi-objective framework for acquisition function optimization, dynamically balancing expected improvement with uncertainty-aware selection. It employs a sparse Gaussian process surrogate model with adaptive noise handling, uses an adaptive hybrid acquisition function that adjusts exploration vs exploitation based on budget progress, and optimizes the acquisition function using a diverse sampling strategy with local refinement. The algorithm incorporates dynamic adaptation of exploration vs exploitation through a multi-objective optimization approach that evolves throughout the optimization process.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution
import warnings

class EfficientRobustBOAO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Acquisition function parameters
        self.noise_level = 1e-6
        
        # Model parameters
        self.model = None
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with sparse approximation to reduce computational cost
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=1,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _adaptive_acquisition(self, X):
        """Adaptive acquisition function that balances expected improvement and uncertainty"""
        if self.model is None:
            return np.zeros(X.shape[0])
            
        # Get predictions from the GP model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Adaptive exploration parameter based on budget progress
        progress_ratio = self.n_evals / self.budget
        
        # Start with high exploration, decrease as we progress
        xi = 0.1 * (1 - progress_ratio) + 0.001
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        improvement = y_min - mu - xi
        z = improvement / sigma
        
        # Expected Improvement
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Normalize EI to [0, 1] for consistent weighting
        if np.max(ei) != np.min(ei):
            ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei))
        else:
            ei_normalized = np.zeros_like(ei)
            
        # Uncertainty-based component
        uncertainty = 1.0 / (sigma + 1e-8)
        
        # Normalize uncertainty to [0, 1] 
        if np.max(uncertainty) != np.min(uncertainty):
            uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        else:
            uncertainty_normalized = np.zeros_like(uncertainty)
            
        # Dynamic weighting based on progress
        if progress_ratio < 0.3:
            # Early stage: prioritize exploration (EI)
            ei_weight = 0.7
            unc_weight = 0.3
        elif progress_ratio < 0.7:
            # Middle stage: balanced
            ei_weight = 0.5
            unc_weight = 0.5
        else:
            # Late stage: prioritize exploitation (uncertainty)
            ei_weight = 0.3
            unc_weight = 0.7
            
        # Combined acquisition function
        acq_value = ei_weight * ei_normalized + unc_weight * uncertainty_normalized
        
        return acq_value

    def _select_next_points(self):
        """Select next points using adaptive acquisition function"""
        # Determine population size based on budget and progress
        if self.n_evals < self.budget * 0.2:
            pop_size = min(50, 10 * self.dim)
        elif self.n_evals < self.budget * 0.6:
            pop_size = min(30, 5 * self.dim)
        else:
            pop_size = min(20, 3 * self.dim)
            
        # For acquisition optimization, use a hybrid approach
        # Generate diverse population with some based on best points
        if self.X is not None and len(self.X) > 0:
            # Use best points as starting locations with some diversity
            best_indices = np.argsort(self.y.flatten())[:min(3, len(self.y))]
            start_points = self.X[best_indices]
            
            # Generate diverse initial population
            population = []
            for i in range(pop_size):
                if i < len(start_points):
                    point = start_points[i % len(start_points)].copy()
                    # Add noise to encourage exploration around best points
                    noise_scale = 0.1 * (self.bounds[1] - self.bounds[0])
                else:
                    point = self._sample_points(1)[0]
                    noise_scale = 0.2 * (self.bounds[1] - self.bounds[0])
                
                # Add noise for diversity
                noise = self.rng.normal(0, noise_scale)
                point += noise
                
                # Clip to bounds
                point = np.clip(point, self.bounds[0], self.bounds[1])
                population.append(point)
        else:
            # No previous points, generate random population
            population = self._sample_points(pop_size)
        
        # Evaluate acquisition function for the entire population
        population_array = np.array(population)
        acq_values = self._adaptive_acquisition(population_array)
        
        # Select top candidates based on acquisition values
        sorted_indices = np.argsort(acq_values)[::-1]
        selected_points = np.array([population[i] for i in sorted_indices[:self.batch_size]])
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate all points in X
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y_vals
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Select next points using acquisition function
            next_points = self._select_next_points()
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

# Import norm for the acquisition function
from scipy.stats import norm

```
The algorithm EfficientRobustBOAO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3500 with standard deviation 0.0117.

took 22.52 seconds to run.

Replace the surrogate model of the selected solution by a substantially different alternative (options: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y, random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted only near the best point), and adapt the rest of the algorithm where needed. Explain in the justification why the new surrogate model should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

