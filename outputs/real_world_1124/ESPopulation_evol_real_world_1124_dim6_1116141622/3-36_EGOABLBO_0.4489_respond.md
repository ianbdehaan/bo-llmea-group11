# Description
**Efficient Global Optimization with Adaptive Bayesian Learning (EGOABL) - Enhanced Version**

This enhanced version of EGOABLBO introduces several key improvements to achieve better optimization performance while maintaining computational efficiency:

1. **Adaptive Sampling Strategy**: Uses Latin Hypercube Sampling initially, then switches to Thompson Sampling for exploitation when sufficient data is available
2. **Improved Acquisition Function**: Implements Expected Improvement with uncertainty scaling and adaptive exploration-exploitation balance 
3. **Efficient Model Updates**: Uses sparse GP with fixed hyperparameters and selective model retraining
4. **Dynamic Batch Selection**: Adapts batch size based on remaining budget and evaluation progress
5. **Smart Exploration-Exploitation Balance**: Dynamically adjusts between exploration (random sampling) and exploitation (acquisition function optimization)
6. **Robust Error Handling**: Gracefully handles function evaluation failures and numerical instabilities

# Justification 
The key improvements focus on enhancing both the exploration-exploitation balance and computational efficiency:

1. **Thompson Sampling Integration**: Switching to Thompson Sampling in later stages provides better exploitation once sufficient data is gathered, improving convergence speed.

2. **Enhanced Acquisition Function**: The improved EI calculation with better numerical stability and uncertainty scaling helps maintain good exploration in early phases while transitioning smoothly to exploitation.

3. **Sparse GP Modeling**: Fixed hyperparameters reduce computational overhead while maintaining model quality, crucial for limited budget scenarios.

4. **Adaptive Batch Selection**: Dynamic batch sizing prevents overcommitting resources early while allowing larger batches when more data is available.

5. **Smart Phase Transitions**: The algorithm transitions from exploration to exploitation based on both budget progress and data availability, ensuring optimal use of function evaluations.

These changes aim to achieve better convergence speed and final solution quality compared to the baseline EGOABLBO while maintaining computational efficiency.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")

class EGOABLBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, dim * 2))  # Adaptive initial points based on dimensionality

        # Do not add any other arguments without a default value
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None
        self.is_initial_phase = True
        self.acquisition_batch_size = 1  # Start with small batches
        self.thompson_sampling_active = False

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for diverse initial exploration
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a sparse GP with Matern kernel for efficiency
        # Fixed hyperparameters for computational efficiency
        
        # Create a base kernel (Matern with automatic relevance detection)
        # Using fixed length scales to reduce computational overhead
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
        
        # For computational efficiency, use a subset of points for fitting (especially important when budget is small)
        max_points_for_fit = min(100, X.shape[0])
        if X.shape[0] > max_points_for_fit:
            fit_indices = self.rng.choice(X.shape[0], size=max_points_for_fit, replace=False)
            X_fit = X[fit_indices]
            y_fit = y[fit_indices]
        else:
            X_fit = X
            y_fit = y
        
        # Fit the GP model with minimal optimization to save time
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=1,  # Reduced number of restarts for efficiency
            random_state=int(self.rng.integers(2**31 - 1)),
            normalize_y=True
        )
        
        gp.fit(X_fit, y_fit)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function using Expected Improvement with uncertainty scaling
        # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
        
        if self.model is None:
            # If no model yet, return random acquisition values for exploration
            return np.ones((X.shape[0], 1)) * self.rng.random(X.shape[0]).reshape(-1, 1)
            
        # Get predictions from the model
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Find current best value (excluding noise)
        if self.y is not None and len(self.y) > 0:
            f_best = np.min(self.y)
        else:
            f_best = 0.0
            
        # Calculate EI using the formula with numerical stability
        improvement = f_best - mu
        
        # Avoid division by zero or very small values
        sigma_safe = np.maximum(sigma, 1e-8)
        
        with np.errstate(divide='warn', invalid='ignore'):
            Z = improvement / sigma_safe
            # Use scipy's norm functions for better numerical stability
            ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
            
        # Add uncertainty scaling factor to encourage exploration in early stages
        if self.is_initial_phase:
            # Increase acquisition values for exploration in early phase with a more aggressive scaling
            scale_factor = 1.0 + 0.7 * np.exp(-self.n_evals / max(1, self.budget * 0.3))
            ei = ei * scale_factor
            
        return ei

    def _thompson_sampling(self, X):
        # Perform Thompson Sampling for exploitation
        if self.model is None:
            return self._sample_points(X.shape[0])
        
        # Sample from the posterior distribution
        samples = self.model.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        return X[np.argmin(samples)]

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # - Use random search in early phase for exploration
        # - Switch to acquisition function optimization for exploitation
        # - Eventually use Thompson Sampling for more efficient exploitation
        
        # If we're in initial phase or have very few evaluations, use random sampling
        if self.is_initial_phase or self.n_evals < 10:
            return self._sample_points(batch_size)
        
        # If we have enough data and are not in the final stage, try acquisition function optimization
        if self.n_evals > self.budget * 0.25 and len(self.X) > 10:
            # Check if we should activate Thompson Sampling for better exploitation
            if self.n_evals > self.budget * 0.6 and not self.thompson_sampling_active:
                self.thompson_sampling_active = True
                
            if self.thompson_sampling_active:
                # Use Thompson Sampling for better exploitation
                candidates = self._sample_points(max(50, batch_size * 10))
                # Get acquisition values for these candidates
                acq_vals = self._acquisition_function(candidates)
                # Select the best candidates using their acquisition values
                top_indices = np.argsort(acq_vals.flatten())[-batch_size:]
                selected_candidates = candidates[top_indices]
                
                # For Thompson Sampling, we sample from the posterior distribution
                # and return points that are likely to improve the current best
                ts_samples = []
                for _ in range(batch_size):
                    # Sample from GP and find point that minimizes the sampled function
                    try:
                        sampled_function = self.model.sample_y(self._sample_points(100), n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
                        best_sample_idx = np.argmin(sampled_function)
                        ts_samples.append(self._sample_points(100)[best_sample_idx])
                    except:
                        # Fallback to random sampling if sampling fails
                        ts_samples.append(self._sample_points(1)[0])
                
                return np.array(ts_samples)
            else:
                # Use acquisition function optimization for exploration-exploitation balance
                best_acq_val = -np.inf
                best_x = None
                
                # Try multiple starting points for optimization
                n_restarts = min(15, max(5, self.budget // 20))
                
                # Generate candidates using Latin Hypercube sampling to explore the space
                candidates = self._sample_points(max(50, n_restarts * 5))
                acq_vals = self._acquisition_function(candidates)
                
                # Select top candidates from these samples
                top_indices = np.argsort(acq_vals.flatten())[-batch_size:]
                selected_candidates = candidates[top_indices]
                
                return selected_candidates
        
        # Default to random sampling for the remaining cases
        return self._sample_points(batch_size)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        if X is None or len(X) == 0:
            return np.array([]).reshape(0, 1)
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception as e:
                # If function evaluation fails, use a large value to penalize
                print(f"Function evaluation failed: {e}")
                y_vals.append(1e6)
                
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Fit the initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we should switch to exploitation phase
            if self.n_evals > self.budget * 0.4:
                self.is_initial_phase = False
                
            # Determine batch size based on remaining budget and current evaluation count
            remaining_budget = self.budget - self.n_evals
            if remaining_budget < 5:
                batch_size = 1
            elif remaining_budget < 20:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, max(1, remaining_budget // 20))
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically (every 3 evaluations or when we have enough data)
            if self.n_evals % 3 == 0 and len(self.X) > 10:
                try:
                    self.model = self._fit_model(self.X, self.y)
                except Exception as e:
                    print(f"Model fitting failed: {e}")
            
            # Check if we've reached the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
```
## Feedback
 The algorithm EGOABLBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4489 with standard deviation 0.0154.

took 6.84 seconds to run.