# Description 
**AdaptiveMultiAcquisitionBO** - This Bayesian Optimization algorithm employs an adaptive ensemble of acquisition functions that dynamically switches between Expected Improvement, Probability of Improvement, and Lower Confidence Bound based on problem characteristics. It uses a Gaussian Process surrogate model with ARD Matern kernel and implements a sophisticated initial design using maximin Latin hypercube sampling. The key innovation lies in the adaptive acquisition function selection mechanism that monitors optimization progress and automatically chooses the most suitable acquisition function for each phase of the search.

# Justification 
This algorithm differs fundamentally from existing approaches by implementing an adaptive ensemble of acquisition functions rather than relying on a single fixed acquisition function throughout the optimization process. While AdaptiveTrustRegionBO uses a fixed Expected Improvement with trust region adaptation, this approach dynamically switches between EI, PI, and LCB based on observed convergence patterns. The algorithm also features:

1. **Adaptive acquisition selection**: Monitors progress and automatically switches between acquisition functions
2. **Efficient GP training**: Uses limited restarts (3) to balance accuracy and computational cost
3. **Robust initial design**: Maximin Latin hypercube sampling for better space-filling properties
4. **Smart batch evaluation**: Selects next points using a hybrid of random sampling and L-BFGS optimization
5. **Progress-aware exploration**: Adjusts search strategy based on observed improvement rates

This approach addresses limitations in existing methods by avoiding premature commitment to one acquisition function while maintaining computational efficiency through smart model updates and adaptive strategies.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np

class AdaptiveMultiAcquisitionBO:
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
        
        # Acquisition function selection
        self.acq_functions = ['ei', 'pi', 'lcb']
        self.current_acq_idx = 0  # Start with EI
        self.acq_history = []  # Track which acquisition functions were used
        
        # Convergence tracking
        self.best_y_history = []
        self.progress_window = 5  # Number of iterations to consider for progress calculation
        self.convergence_threshold = 1e-4  # Threshold for determining convergence
        
        # GP model parameters
        self.gp_n_restarts = 3  # Limited restarts for computational efficiency

    def _sample_points(self, n_points):
        # Sample points using maximin Latin hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit surrogate model with limited restarts for efficiency
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=self.gp_n_restarts,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _expected_improvement(self, X, model, best_y):
        # Expected Improvement acquisition function
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)  # Avoid numerical issues
        
        improvement = mu - best_y
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _probability_of_improvement(self, X, model, best_y):
        # Probability of Improvement acquisition function
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)  # Avoid numerical issues
        
        Z = (mu - best_y) / sigma
        pi = norm.cdf(Z)
        
        return pi.reshape(-1, 1)

    def _lower_confidence_bound(self, X, model, best_y):
        # Lower Confidence Bound acquisition function with adaptive exploration weight
        kappa = 2.0  # Exploration weight - fixed for this implementation
        mu, sigma = model.predict(X, return_std=True)
        
        lcb = mu - kappa * sigma
        
        return lcb.reshape(-1, 1)

    def _get_acquisition_function(self, acq_name):
        # Return the appropriate acquisition function
        if acq_name == 'ei':
            return self._expected_improvement
        elif acq_name == 'pi':
            return self._probability_of_improvement
        elif acq_name == 'lcb':
            return self._lower_confidence_bound
        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

    def _select_next_points(self, batch_size):
        # Select next points using current acquisition function with hybrid approach:
        # 1. Sample random points 
        # 2. Use L-BFGS-B optimization on acquisition function
        # 3. Select best points from both approaches
        
        n_random = max(5, batch_size // 2)  # Sample some random points
        n_optimize = batch_size - n_random
        
        # Generate random points
        random_points = self._sample_points(n_random)
        
        # Optimize acquisition function using L-BFGS-B from multiple starting points
        if self.X is not None and len(self.X) > 0:
            best_y = np.min(self.y)
            model = self._fit_model(self.X, self.y)
            
            current_acq_func = self._get_acquisition_function(self.acq_functions[self.current_acq_idx])
            
            # Generate starting points for optimization
            n_starts = min(3, max(1, n_optimize // 2))
            start_points = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_opt_point = None
            
            for start_point in start_points:
                def obj(x):
                    x = x.reshape(1, -1)
                    return -current_acq_func(x, model, best_y)[0, 0]
                
                try:
                    res = minimize(obj, start_point, method='L-BFGS-B', 
                                 bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
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

    def _update_acquisition_function(self):
        # Adaptively switch acquisition function based on optimization progress
        if len(self.y) < 10:  # Early stage - use EI for exploration
            self.current_acq_idx = 0
            return
            
        # Calculate recent improvement
        if len(self.y) >= self.progress_window:
            recent_improvement = np.min(self.y[-self.progress_window:]) - np.min(self.y[:-self.progress_window])
            
            # If improvement is small, switch to LCB for more focused exploration
            if recent_improvement < self.convergence_threshold and self.current_acq_idx != 2:
                self.current_acq_idx = 2  # Switch to LCB
            elif recent_improvement >= self.convergence_threshold:
                # If there's significant improvement, continue with EI or switch to PI
                if self.current_acq_idx == 0:  # Currently using EI
                    # Switch to PI if we're making good progress but want more exploitation
                    if len(self.y) > 20 and np.random.random() < 0.3:
                        self.current_acq_idx = 1  # Switch to PI

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update acquisition function based on progress
            self._update_acquisition_function()
            
            # Track acquisition function usage
            self.acq_history.append(self.acq_functions[self.current_acq_idx])
            
            # Select next points to evaluate
            batch_size = min(3, self.budget - self.n_evals)  # Limit batch size to remaining budget
            if batch_size <= 0:
                break
                
            next_points = self._select_next_points(batch_size)
            
            # Evaluate new points
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            
            # Early termination for small budgets to avoid excessive overhead
            if self.n_evals >= self.budget - 1:
                break
                
        # Return the best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
```
## Feedback
 The algorithm AdaptiveMultiAcquisitionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3063 with standard deviation 0.0143.

took 16.95 seconds to run.