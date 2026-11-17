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
        self.n_init = min(10, max(2 * dim + 1, 10)) # Initial design size
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Surrogate model parameters
        self.gp_model = None
        self.rf_model = None
        
        # Acquisition function parameters
        self.acq_func_type = "ei"  # ei, lcb, thompson
        self.lcb_weight = 0.1

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for initial design
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit GP model with reduced hyperparameter optimization restarts
        try:
            # Use a more robust initialization for GP
            kernel = Matern(length_scale=np.ones(self.dim), length_scale_bounds=[(0.1, 10)]*self.dim, nu=2.5)
            # Add WhiteKernel for noise modeling
            kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
            
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=1,  # Reduced from 3 to 1 for speed
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            self.gp_model.fit(X, y)
        except Exception as e:
            warnings.warn(f"GP fitting failed: {e}")
            self.gp_model = None

    def _acquisition_function_ei(self, X):
        """Expected Improvement acquisition function"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        try:
            # Expected Improvement with GP model
            mean, std = self.gp_model.predict(X, return_std=True)
            best_y = np.min(self.y)
            improvement = best_y - mean
            # EI formula
            Z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei = np.clip(ei, -1e10, 1e10)
            return ei.reshape(-1, 1)
        except Exception as e:
            warnings.warn(f"GP EI calculation failed: {e}")
            return np.zeros((X.shape[0], 1))

    def _acquisition_function_lcb(self, X):
        """Lower Confidence Bound acquisition function"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        try:
            # LCB with GP model
            mean, std = self.gp_model.predict(X, return_std=True)
            lcb = mean - self.lcb_weight * std
            return lcb.reshape(-1, 1)
        except Exception as e:
            warnings.warn(f"GP LCB calculation failed: {e}")
            return np.zeros((X.shape[0], 1))

    def _optimize_acquisition(self, n_restarts=5):
        """Optimize acquisition function using multiple random restarts"""
        if self.X is None or len(self.X) < 2:
            # For initial points, use random sampling
            return self._sample_points(1)[0]
        
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple random starting points
        for _ in range(n_restarts):
            # Random starting point within bounds
            x_start = self.bounds[0] + self.rng.random(self.dim) * (self.bounds[1] - self.bounds[0])
            
            try:
                # Use L-BFGS-B optimizer for acquisition function
                from scipy.optimize import minimize
                
                # Optimize with different acquisition functions
                if self.acq_func_type == "ei":
                    def acq_func(x):
                        x = x.reshape(1, -1)
                        return -self._acquisition_function_ei(x)[0, 0]
                else:  # lcb
                    def acq_func(x):
                        x = x.reshape(1, -1)
                        return -self._acquisition_function_lcb(x)[0, 0]
                
                result = minimize(
                    acq_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 20}
                )
                
                if result.success:
                    acq_val = -result.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_x = result.x
                        
            except Exception as e:
                warnings.warn(f"Acquisition optimization failed: {e}")
                continue
        
        # If no good point found, return a random one
        if best_x is None:
            return self._sample_points(1)[0]
            
        return best_x

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
        
        # Iteratively optimize - use fixed batch size of 1 for budget predictability
        batch_size = 1
        
        # Adaptive strategy: switch between EI and LCB based on early progress
        early_improvement_threshold = 0.5  # If improvement is less than this, switch to LCB
        improvement_window = 20  # Number of evaluations to look back
        
        while self.n_evals < self.budget - batch_size:
            if self.X is None or len(self.X) < 2:
                break
                
            # Fit models after each batch
            self._fit_model(self.X, self.y)
            
            # Optimize acquisition function for next point
            next_point = self._optimize_acquisition(n_restarts=3)
            
            # Evaluate point
            next_values = self._evaluate_points(func, next_point.reshape(1, -1))
            
            # Update points
            self._update_eval_points(next_point.reshape(1, -1), next_values)
            
            # Adaptive acquisition function switching based on recent improvement
            if len(self.y) > improvement_window:
                recent_improvement = np.min(self.y[-improvement_window:]) - np.min(self.y)
                if recent_improvement < early_improvement_threshold:
                    self.acq_func_type = "lcb"
                else:
                    # Occasionally switch back to EI
                    if self.rng.random() < 0.1:
                        self.acq_func_type = "ei"
        
        # Final evaluation of best point
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
