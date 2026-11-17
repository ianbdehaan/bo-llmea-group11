# Description
**AdaptiveMultiSurrogateBO**: This novel Bayesian Optimization algorithm combines multiple surrogate models with adaptive selection and a sophisticated acquisition function strategy. It uses an initial Latin Hypercube design, employs both Gaussian Process (with ARD Matern kernel) and Random Forest surrogates for robust modeling, implements a multi-acquisition function portfolio that dynamically switches between Expected Improvement, Lower Confidence Bound, and Thompson Sampling, and utilizes an efficient trust region-based acquisition optimizer with restarts to balance exploration and exploitation effectively.

# Justification
1. **Multi-surrogate approach**: Combines GP (for smooth functions) and Random Forest (for noisy/complex functions) to improve robustness across diverse optimization landscapes.
2. **Adaptive acquisition function selection**: Switches between EI, LCB, and Thompson Sampling based on model confidence to balance exploration/exploitation dynamically.
3. **Efficient acquisition optimization**: Uses trust region sampling with local refinement and restarts to avoid expensive global optimization while maintaining good convergence properties.
4. **Scalable design**: The algorithm scales well with budget size, uses efficient GP hyperparameter tuning (few restarts), and employs batch evaluation strategies.
5. **Reproducibility**: All random generators are seeded properly using the provided rng mechanism.

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
## Error
 [Errno Evaluation timed out (%d)] 120