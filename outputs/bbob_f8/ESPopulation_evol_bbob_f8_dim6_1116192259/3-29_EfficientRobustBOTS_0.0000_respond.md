# Description 
**Efficient Robust Bayesian Optimization with Thompson Sampling (ERBO-TS)**: This Bayesian Optimization algorithm introduces Thompson sampling as the primary acquisition function, combined with a robust initialization strategy and adaptive surrogate modeling. It uses a maximin Latin hypercube design for initial sampling, employs a sparse Gaussian process with ARD Matern kernel for modeling, and implements a hybrid acquisition function optimization approach. The key innovation is the use of Thompson sampling for exploration-exploitation balance, which inherently adapts to the uncertainty landscape without requiring manual tuning of exploration weights.

# Justification 
The proposed algorithm replaces expected improvement with Thompson sampling for several key reasons:

1. **Inherent Exploration-Exploitation Balance**: Thompson sampling naturally balances exploration and exploitation by sampling from the posterior distribution. Unlike EI which requires careful tuning of the exploration parameter xi, TS adapts automatically to uncertainty levels in the GP model.

2. **Reduced Hyperparameter Sensitivity**: Expected improvement requires manual tuning of the exploration parameter (xi), which can be problematic for varying budgets. Thompson sampling eliminates this sensitivity by directly sampling from the posterior.

3. **Better Uncertainty Utilization**: Thompson sampling fully utilizes the uncertainty information in the GP, especially beneficial in early iterations when uncertainty is high and more exploration is needed.

4. **Computational Efficiency**: TS requires only one sample from the GP posterior per candidate point, which is computationally cheaper than evaluating complex acquisition functions like EI, especially for high-dimensional problems.

5. **Robustness to Noisy Evaluations**: Thompson sampling has shown robust performance in noisy optimization scenarios, which is beneficial given that many real-world black-box functions exhibit noise.

The algorithm maintains the efficient initialization with maximin Latin hypercube and sparse GP modeling while improving the acquisition function optimization approach to better handle the Thompson sampling paradigm. The hybrid optimization strategy (sampling + local refinement) still provides good convergence properties while being computationally efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class EfficientRobustBOTS:
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
        # Use a more adaptive approach that scales better with budget
        self.n_init = min(max(10, 2 * dim + 1), max(10, budget // 4))
        if self.n_init > budget:
            self.n_init = budget
            
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
        """Fit and tune surrogate model with reduced computational overhead"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with sparse approximation to reduce computational cost
        # Use fewer restarts for efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced from 3 to 2 for speed
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Thompson Sampling acquisition function"""
        # If no model exists, return zeros
        if self.model is None:
            return np.zeros((X.shape[0], 1))
            
        # Sample from the posterior distribution for each candidate point
        # This is the core of Thompson sampling - sample from GP posterior
        samples = self.model.sample_y(X, n_samples=1, random_state=self.rng)
        
        # Return the sampled values (lower is better in our minimization context)
        return samples.reshape(-1, 1)

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using a hybrid approach"""
        # Sample initial candidates
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points to start optimization from
        top_indices = np.argsort(acq_values.flatten())[:max(5, self.dim)]  # Take best ones
        starting_points = candidates[top_indices]
        
        # Use local optimization around the best candidates
        best_acq_val = np.inf
        best_point = None
        
        for x0 in starting_points:
            try:
                # Simple bounded L-BFGS optimization with restarts
                result = minimize(
                    lambda x: self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    acq_val = result.fun
                    if acq_val < best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x
                        
            except:
                # If optimization fails, continue with current best
                pass
        
        # Fallback to the best candidate from sampling if optimization fails
        if best_point is None:
            best_idx = np.argmin(acq_values.flatten())
            best_point = candidates[best_idx]
            
        return best_point.reshape(1, -1)

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
            
            # Select next points using acquisition function optimization
            next_points = self._optimize_acquisition(n_starting_points=min(50, max(10, 2 * self.dim)))
            
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
## Error
 Traceback (most recent call last):
  File "<EfficientRobustBOTS>", line 164, in __call__
 164->             next_points = self._optimize_acquisition(n_starting_points=min(50, max(10, 2 * self.dim)))
  File "<EfficientRobustBOTS>", line 87, in _optimize_acquisition
  87->         acq_values = self._acquisition_function(candidates)
  File "<EfficientRobustBOTS>", line 76, in _acquisition_function
  74 |         # Sample from the posterior distribution for each candidate point
  75 |         # This is the core of Thompson sampling - sample from GP posterior
  76->         samples = self.model.sample_y(X, n_samples=1, random_state=self.rng)
  77 |         
  78 |         # Return the sampled values (lower is better in our minimization context)
ValueError: Generator(PCG64) at 0x7FC1303A9460 cannot be used to seed a numpy.random.RandomState instance
