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
- EfficientRobustBO: 0.3452, 214.65 seconds, **Efficient Robust Bayesian Optimization (ERBO)**: This algorithm combines a robust initialization strategy with an adaptive surrogate model and acquisition function to efficiently navigate high-dimensional black-box optimization. It uses a maximin Latin hypercube design for initial sampling, employs a sparse Gaussian process with ARD Matern kernel for modeling, and implements a multi-start L-BFGS optimizer for acquisition function maximization. The algorithm incorporates uncertainty-aware exploration through a novel acquisition function that balances global search with local exploitation.


- AdaptiveTrustRegionBO: 0.3452, 31.65 seconds, **AdaptiveTrustRegionBO** - This Bayesian Optimization algorithm combines an adaptive trust region mechanism with a multi-start acquisition function optimizer. It uses a Gaussian Process surrogate model with ARD Matern kernel and employs expected improvement acquisition function. The key innovation is the dynamic trust region that adapts based on function behavior, allowing efficient exploration in early stages and focused exploitation near the optimum. The algorithm uses a hybrid approach for acquisition optimization: random sampling with local refinement using L-BFGS-B. It incorporates a restart mechanism when progress stagnates and uses a sophisticated initial design strategy.


- EfficientRobustBOES: 0.3417, 30.39 seconds, **Efficient Robust Bayesian Optimization with Evolution Strategy (ERBO-ES)**: This algorithm combines a robust initialization strategy with an adaptive surrogate model and acquisition function to efficiently navigate high-dimensional black-box optimization. It uses a maximin Latin hypercube design for initial sampling, employs a sparse Gaussian process with ARD Matern kernel for modeling, and implements a small evolution strategy optimizer for acquisition function maximization. The algorithm incorporates uncertainty-aware exploration through a novel acquisition function that balances global search with local exploitation.


- AdaptiveMultiSurrogateBO: 0.3416, 49.78 seconds, **AdaptiveMultiSurrogateBO**: This Bayesian Optimization algorithm employs an adaptive surrogate model selection strategy that dynamically switches between Gaussian Process (GP) and Random Forest (RF) models based on performance metrics. It uses a novel hybrid acquisition function combining Expected Improvement with a local exploration term, optimized via a multi-start L-BFGS approach with intelligent restarts. The algorithm implements an efficient initialization using a maximin Latin Hypercube design and employs a progressive refinement strategy that adapts the search space based on observed improvements.




The selected solution to update is:
**AdaptiveMultiSurrogateBO**: This Bayesian Optimization algorithm employs an adaptive surrogate model selection strategy that dynamically switches between Gaussian Process (GP) and Random Forest (RF) models based on performance metrics. It uses a novel hybrid acquisition function combining Expected Improvement with a local exploration term, optimized via a multi-start L-BFGS approach with intelligent restarts. The algorithm implements an efficient initialization using a maximin Latin Hypercube design and employs a progressive refinement strategy that adapts the search space based on observed improvements.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
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
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Model parameters
        self.gp_model = None
        self.rf_model = None
        self.current_model = "gp"  # Start with GP model
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for surrogate switching
        self.gp_performance = []
        self.rf_performance = []
        
        # Exploration parameters
        self.exploration_weight = 0.01
        self.min_exploration = 0.001

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _fit_rf_model(self, X, y):
        """Fit Random Forest surrogate model"""
        # Use RF with good default parameters for robustness
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single thread to avoid parallelization issues
        )
        
        model.fit(X, y)
        return model

    def _predict_with_model(self, model, X):
        """Predict with either GP or RF model"""
        if hasattr(model, 'predict'):
            try:
                # For GP model
                mu, sigma = model.predict(X, return_std=True)
                return mu, sigma
            except:
                # Fallback to RF prediction
                return model.predict(X), np.zeros(X.shape[0])
        else:
            # RF model
            return model.predict(X), np.zeros(X.shape[0])

    def _calculate_performance(self, model, X, y):
        """Calculate mean squared error for performance metrics"""
        try:
            predictions = model.predict(X)
            mse = np.mean((y.flatten() - predictions)**2)
            return mse
        except:
            return float('inf')

    def _switch_surrogate_model(self):
        """Dynamically switch between GP and RF models based on performance"""
        if self.n_evals < 20:  # Wait until we have enough data
            return
            
        # Calculate current performance for both models
        gp_performance = self._calculate_performance(self.gp_model, self.X, self.y)
        rf_performance = self._calculate_performance(self.rf_model, self.X, self.y)
        
        # Track performances
        self.gp_performance.append(gp_performance)
        self.rf_performance.append(rf_performance)
        
        # Switch if RF performs significantly better (threshold of 20% improvement)
        if len(self.rf_performance) >= 3 and len(self.gp_performance) >= 3:
            recent_gp = np.mean(self.gp_performance[-3:])
            recent_rf = np.mean(self.rf_performance[-3:])
            
            if recent_rf < 0.8 * recent_gp:
                self.current_model = "rf"
            elif recent_gp < 0.8 * recent_rf:
                self.current_model = "gp"

    def _acquisition_function(self, X):
        """Hybrid acquisition function combining Expected Improvement with local exploration"""
        if self.gp_model is None and self.rf_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Select model based on current state
        if self.current_model == "gp":
            model = self.gp_model
        else:
            model = self.rf_model
            
        # Get predictions from the selected model
        mu, sigma = self._predict_with_model(model, X)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Adaptive exploration weight based on progress
        progress_ratio = self.n_evals / self.budget
        # Decrease exploration as we progress through budget (more exploitation, less exploration)
        xi = self.exploration_weight * (1 - progress_ratio) + self.min_exploration
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add local exploration term (this helps escape local optima)
        # Use a simple distance-based exploration penalty
        if self.X is not None and len(self.X) > 0:
            # Compute distances from all existing points
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            
            # Local exploration term - encourage points far from existing evaluations
            local_exploration = 1.0 / (min_distances + 1e-8)  # Avoid division by zero
            
            # Combine with EI
            ei = ei.reshape(-1, 1) + 0.1 * local_exploration.reshape(-1, 1)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Initialize with a good set of starting points
        n_starting_points = min(50 * self.dim, 1000)  # Reduced from previous version for efficiency
        
        # Sample initial candidates using Latin hypercube for better coverage
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates (use a simple heuristic - just the best ones)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
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
        
        # Fit initial models
        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.rf_model = self._fit_rf_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data (both)
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
                self.rf_model = self._fit_rf_model(self.X, self.y)
                
                # Switch surrogate model based on performance
                self._switch_surrogate_model()
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
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

```
The algorithm AdaptiveMultiSurrogateBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3416 with standard deviation 0.0104.

took 49.78 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

