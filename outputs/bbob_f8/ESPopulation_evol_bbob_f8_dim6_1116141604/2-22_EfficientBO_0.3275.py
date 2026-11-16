from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EfficientBO:
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
        self.acq_n_restarts = min(20, max(5, budget // 10))
        self.acq_batch_size = 1
        
        # Smart budget allocation parameters
        self.adaptive_batch_size = True
        self.model_update_frequency = max(1, budget // 20)
        self.switch_threshold = 0.9  # Switch to linear if linear error < gp_error * threshold

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
                    n_restarts_optimizer=min(5, max(1, self.n_evals // 10)),
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
        """Compute hybrid acquisition function combining EI and Uncertainty Sampling"""
        if self.model_type == "gp":
            return self._compute_hybrid_acq_gp(X)
        else:
            return self._compute_hybrid_acq_linear(X)

    def _compute_hybrid_acq_gp(self, X):
        """Compute hybrid acquisition for Gaussian Process model"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value
            y_best = np.min(self.y)
            
            # Compute Expected Improvement
            improvement = y_best - mean - 1e-8  # Small epsilon to prevent division by zero
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Compute Uncertainty Sampling (inverse of std)
            uncertainty = 1.0 / (std + 1e-8)
            
            # Combine both for hybrid acquisition
            # Weighting factor: higher weight to EI in later stages, more to uncertainty in early stages
            if self.n_evals > self.budget * 0.5:
                ei_weight = 0.7
                uncertainty_weight = 0.3
            else:
                ei_weight = 0.4
                uncertainty_weight = 0.6
                
            hybrid_acq = ei_weight * ei + uncertainty_weight * uncertainty
            
            return hybrid_acq.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_hybrid_acq_linear(self, X):
        """Compute hybrid acquisition for Linear model"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic (this is not true EI but maintains exploration)
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            # Add uncertainty component based on variance of training data
            y_variance = np.var(self.y) if len(self.y) > 1 else 1.0
            uncertainty = np.ones((X.shape[0], 1)) * y_variance
            
            # Hybrid approach: combine EI with uncertainty
            hybrid_acq = 0.7 * ei + 0.3 * uncertainty
            
            return hybrid_acq
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Use a hybrid approach for acquisition optimization
        best_acq_val = -np.inf
        best_point = None
        
        # Try multiple random starts to avoid local optima
        n_starts = min(30, max(5, self.acq_n_restarts))
        
        # Sample more points for better exploration in early stages
        if self.n_evals < self.budget * 0.3:
            sample_size = 20
        else:
            sample_size = 10
            
        for _ in range(n_starts):
            # Sample initial points for optimization
            start_points = self._sample_points(sample_size)
            
            # Try to optimize acquisition function from these points
            try:
                acq_vals = self._acquisition_function(start_points)
                
                # Select the point with highest acquisition value
                max_idx = np.argmax(acq_vals.flatten())
                if acq_vals[max_idx] > best_acq_val:
                    best_acq_val = acq_vals[max_idx]
                    best_point = start_points[max_idx].reshape(1, -1)
            except:
                continue
        
        if best_point is None:
            # Fallback to random sampling
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

    def _should_update_model(self):
        """Determine if we should update the model based on budget and evaluation count"""
        # Update model every few evaluations or when we have enough data
        return (self.n_evals % self.model_update_frequency == 0) and (len(self.X) > 5)

    def _adaptive_batch_size(self):
        """Dynamically adjust batch size based on remaining budget"""
        if self.budget <= 100:
            return 1
        elif self.budget <= 300:
            # For medium budgets, use larger batches in early stages
            if self.n_evals < self.budget * 0.4:
                return min(5, max(1, self.budget // 50))
            else:
                return 1
        else:
            # For large budgets, batch size can be larger initially
            if self.n_evals < self.budget * 0.3:
                return min(10, max(1, self.budget // 40))
            elif self.n_evals < self.budget * 0.7:
                return min(5, max(1, self.budget // 100))
            else:
                return 1

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
        while self.n_evals < self.budget:
            # Check if we have enough evaluations to continue
            if self.n_evals >= self.budget:
                break
                
            # Determine batch size
            batch_size = self._adaptive_batch_size()
            
            # Select next points for evaluation
            X_next = self._select_next_points(batch_size)
            
            # Evaluate next point(s)
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows
            if self._should_update_model() and len(self.X) > 5:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance
            if len(self.X) > 15 and self.n_evals < self.budget * 0.8:
                try:
                    # Test if GP is significantly better than linear
                    if hasattr(self.gp_model, 'predict'):
                        gp_pred = self.gp_model.predict(self.X, return_std=False)
                        if self.linear_model is not None:
                            lin_pred = self.linear_model.predict(self.scaler.transform(self.X)) 
                            if len(lin_pred) == len(self.y):
                                gp_error = mean_squared_error(self.y, gp_pred)
                                lin_error = mean_squared_error(self.y, lin_pred.reshape(-1, 1))
                                
                                # Switch to linear if it's better (but not too early)
                                if lin_error < gp_error * self.switch_threshold and self.model_type != "linear":
                                    self.model_type = "linear"
                                    self.linear_model = self._fit_linear_model(self.X, self.y)
                except:
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
