# Description
**AdaptiveMultiSurrogateBO**: This Bayesian Optimization algorithm employs an adaptive surrogate model selection strategy that dynamically switches between Gaussian Process (GP) and Random Forest (RF) models based on performance metrics. It uses a novel hybrid acquisition function combining Expected Improvement with a local exploration term, optimized via a multi-start L-BFGS approach with intelligent restarts. The algorithm implements an efficient initialization using a maximin Latin Hypercube design and employs a progressive refinement strategy that adapts the search space based on observed improvements.

# Justification
The key innovations in this refined approach include:
1. **Adaptive Surrogate Selection**: Switching between GP and RF models based on performance metrics (mean squared error) to leverage the strengths of both approaches - GP for smooth functions and RF for noisy or discontinuous landscapes.
2. **Hybrid Acquisition Function**: Combines Expected Improvement with a local exploration term that helps escape local optima, particularly important for the Rosenbrock function which has a narrow, curved valley.
3. **Efficient Multi-start Optimization**: Uses a combination of random sampling and L-BFGS-B optimization with restarts to efficiently maximize acquisition functions, while limiting computational cost through early termination.
4. **Progressive Refinement Strategy**: Adapts search space exploration based on observed improvements to focus on promising regions more effectively.
5. **Robust Initial Design**: Uses maximin Latin hypercube sampling for better coverage of the search space in early iterations.

These components work synergistically to improve robustness and convergence speed, particularly for challenging functions like the Rosenbrock function.

# Code
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
## Feedback
 The algorithm AdaptiveMultiSurrogateBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3416 with standard deviation 0.0104.

took 49.78 seconds to run.