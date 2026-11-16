from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EfficientBayesianOptimizationBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Set initial points based on budget and dimension
        if budget <= 50:
            self.n_init = max(5, min(10, budget // 2))
        elif budget <= 200:
            self.n_init = max(10, min(20, budget // 4))
        else:
            self.n_init = max(15, min(30, budget // 6))
        
        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model parameters
        self.model_type = "gp"  # Switch between 'gp' and 'linear'
        self.gp_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_n_restarts = min(30, max(10, budget // 8))
        self.acq_batch_size = 1
        self.ei_gamma = 0.01  # Small constant for EI computation stability
        
        # Model update parameters
        self.min_model_updates = 5
        self.model_update_frequency = max(1, budget // 20)
        
        # Exploration-exploitation balance
        self.explore_weight = 0.7
        self.exploit_weight = 0.3

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling"""
        # Use Sobol sampling for initial points
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        """Fit surrogate model with adaptive selection"""
        if self.model_type == "gp":
            # Fit Gaussian Process with Matern kernel
            try:
                # Use a more robust kernel
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=min(5, max(1, self.n_evals // 15)),
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X, y)
                return gp
            except:
                # Fallback to linear model if GP fails
                self.model_type = "linear"
                return self._fit_linear_model(X, y)
        else:
            return self._fit_linear_model(X, y)

    def _fit_linear_model(self, X, y):
        """Fit a simple linear model as fallback"""
        try:
            # Scale the data for numerical stability
            X_scaled = self.scaler.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_scaled, y.flatten())
            return lr
        except:
            # Fallback to constant prediction
            return np.mean(y)

    def _acquisition_function(self, X):
        """Compute Expected Improvement acquisition function"""
        if self.model_type == "gp":
            return self._compute_ei_gp(X)
        else:
            return self._compute_ei_linear(X)

    def _compute_ei_gp(self, X):
        """Compute EI for Gaussian Process model with improved numerical stability"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value with numerical stability
            y_best = np.min(self.y) - self.ei_gamma
            
            # Compute EI with better numerical handling
            improvement = y_best - mean
            # Add small epsilon to avoid numerical issues
            std_safe = std + 1e-8
            
            # Handle cases where std is very small
            z = improvement / std_safe
            ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
            
            # Ensure non-negative EI for better exploration
            ei = np.maximum(ei, 0)
            
            return ei.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ei_linear(self, X):
        """Compute EI for Linear model with improved approach"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a more sophisticated approach
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Calculate improvement based on current best and predictions
            y_best = np.min(self.y) - self.ei_gamma
            
            # Simple improvement heuristic with better handling of negative values
            ei = np.maximum(y_best - predictions, 0)
            
            return ei
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Use a more sophisticated approach for acquisition optimization
        best_acq_val = -np.inf
        best_point = None
        
        # Try multiple random starts to avoid local optima
        n_starts = min(50, max(10, self.acq_n_restarts))
        
        for _ in range(n_starts):
            # Sample initial points for optimization
            start_points = self._sample_points(3)
            
            try:
                # Compute acquisition values for these points
                acq_vals = self._acquisition_function(start_points)
                
                # Select the point with highest acquisition value
                max_idx = np.argmax(acq_vals.flatten())
                if acq_vals[max_idx] > best_acq_val:
                    best_acq_val = acq_vals[max_idx]
                    best_point = start_points[max_idx].reshape(1, -1)
            except:
                continue
        
        # If we couldn't find a good point, fall back to random sampling
        if best_point is None:
            return self._sample_points(batch_size)
            
        return best_point

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate points and return function values
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _evaluate_model_performance(self):
        """Evaluate current model performance for adaptive switching"""
        if self.gp_model is None or self.X is None or self.y is None:
            return False
        
        try:
            # Get predictions
            gp_pred = self.gp_model.predict(self.X, return_std=False)
            
            # Calculate MSE
            mse_gp = mean_squared_error(self.y.flatten(), gp_pred.flatten())
            
            # For linear model comparison
            if self.linear_model is not None:
                lin_pred = self.linear_model.predict(self.scaler.transform(self.X))
                mse_lin = mean_squared_error(self.y.flatten(), lin_pred)
                
                # Switch to linear if it's significantly better
                return mse_lin < mse_gp * 0.85
            else:
                # If no linear model yet, evaluate with a simple heuristic
                return False
                
        except:
            return False

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        # Initialize
        self.X = None
        self.y = None
        self.n_evals = 0
        
        # Generate initial points
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        
        # Store initial points
        self._update_eval_points(X_init, y_init)
        
        # Fit the initial model
        self.gp_model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        iteration = 0
        
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to continue
            if self.n_evals >= self.budget:
                break
                
            # Select next points for evaluation
            X_next = self._select_next_points(1)
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows
            if len(self.X) > self.min_model_updates and (iteration % self.model_update_frequency == 0):
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                    
                    # Check if we should switch models based on performance
                    if len(self.X) > 15 and self.n_evals < self.budget * 0.8:
                        should_switch = self._evaluate_model_performance()
                        if should_switch and self.model_type != "linear":
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            iteration += 1
            
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
