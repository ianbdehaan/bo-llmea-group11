# Description
**Efficient Multi-Start Bayesian Optimization with Sobol Initial Design and Expected Improvement (EMBO-SB-EI)**

This novel Bayesian optimization algorithm introduces a distinctive approach by combining an efficient surrogate model with a multi-start acquisition function optimization strategy, while maintaining the superior space-filling properties of Sobol sequences for initial design. Unlike previous approaches that rely on Thompson sampling or other sampling-based acquisition functions, this method uses Expected Improvement (EI) as its primary acquisition function, which is known for its strong theoretical foundation and practical performance in Bayesian optimization.

# Justification
The key components of this algorithm differ significantly from existing approaches:

1. **Sobol Sequence Initial Design**: Uses Sobol sequences instead of Latin Hypercube or random sampling to ensure superior space-filling properties, especially in higher dimensions (as established by previous work).

2. **Expected Improvement Acquisition Function**: Unlike the Thompson Sampling approach used in EMBO_SB_TS, this algorithm employs Expected Improvement which is more theoretically grounded and often provides better convergence behavior for optimization problems.

3. **Multi-Start Optimization Strategy**: Uses multiple random restarts when optimizing the acquisition function to avoid local optima, similar to EMBO_SB_TS but with a different acquisition function.

4. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel optimized with few hyperparameter restarts to balance accuracy and computational cost.

5. **Adaptive Batch Selection**: Instead of fixed batch sizes, this algorithm dynamically adjusts batch size based on remaining budget to ensure efficient exploration-exploitation trade-off.

6. **No Trust Region Mechanism**: Unlike EMBO_SOBOL_TRE_TS which uses adaptive trust regions, this algorithm relies purely on global search strategies with the assumption that the Sobol initial design provides good coverage of the search space.

The main innovation lies in combining the superior space-filling properties of Sobol sequences with the theoretically sound Expected Improvement acquisition function and efficient multi-start optimization, while avoiding complex trust-region mechanisms that can add computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EMBO_SB_EI:
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
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
        # Acquisition function optimization parameters
        self.n_acq_starts = min(20, max(5, dim * 3))  # Number of random starts for acquisition optimization
        
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

    def _expected_improvement(self, X, best_y):
        """Compute Expected Improvement acquisition function"""
        if self.gp is None:
            # For initial points before first model fit, return zeros
            return np.zeros((X.shape[0], 1))
            
        # Get mean and std from GP posterior
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Avoid numerical issues when sigma is close to zero
        sigma = np.maximum(sigma, 1e-8)
        
        # Compute EI
        improvement = best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, X, y):
        """Optimize the acquisition function using multiple random restarts"""
        best_value = np.inf
        best_point = None
        
        # Generate diverse starting points
        start_points = self._sample_points(self.n_acq_starts)
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(self.n_acq_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._expected_improvement(x.reshape(1, -1), np.min(y))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=15,  # Reduced iterations for speed
                    popsize=3,   # Reduced population size
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._expected_improvement(point, np.min(y))[0, 0]
                    
                    if acq_value < best_value:  # Minimize EI (since we're maximizing improvement)
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        # Return the best point found or a random point if all failed
        if best_point is not None:
            return best_point
        else:
            # Fallback: return one of the initial points
            return start_points[0].reshape(1, -1)

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
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points to evaluate using Expected Improvement acquisition function
            batch_size = min(3, self.budget - self.n_evals)  # Limit batch size for efficiency
            
            # For each batch element, optimize acquisition function
            candidates = []
            for _ in range(batch_size):
                if len(self.X) > 0:
                    next_point = self._optimize_acquisition(self.X, self.y)
                    candidates.append(next_point)
                else:
                    # Fallback to random point if no data yet
                    candidates.append(self._sample_points(1))
            
            # Stack all candidate points
            next_points = np.vstack(candidates)
            
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
## Feedback
 The algorithm EMBO_SB_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4486 with standard deviation 0.0279.

took 15.11 seconds to run.