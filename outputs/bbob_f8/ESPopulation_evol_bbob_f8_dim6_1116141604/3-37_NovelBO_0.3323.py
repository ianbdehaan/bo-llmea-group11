from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

class NovelBO:
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
        self.acq_n_restarts = min(5, max(2, budget // 20))  # Reduced restarts for efficiency
        self.acq_batch_size = 1
        self.epsilon = 1e-8
        
        # Performance tracking for model switching
        self.gp_performance_history = []
        self.linear_performance_history = []
        
        # Optimization parameters
        self.model_update_threshold = 0.8  # Only update model if we have enough evaluations

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
                    n_restarts_optimizer=min(3, max(1, self.n_evals // 20)),  # Reduced for efficiency
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

    def _compute_gp_performance(self, X_test, y_test):
        """Compute performance metric for GP model"""
        if self.gp_model is None:
            return float('inf')
        
        try:
            predictions = self.gp_model.predict(X_test, return_std=False)
            mse = np.mean((predictions.flatten() - y_test.flatten())**2)
            return mse
        except:
            return float('inf')

    def _compute_linear_performance(self, X_test, y_test):
        """Compute performance metric for linear model"""
        if self.linear_model is None:
            return float('inf')
        
        try:
            X_scaled = self.scaler.transform(X_test)
            predictions = self.linear_model.predict(X_scaled)
            mse = np.mean((predictions.flatten() - y_test.flatten())**2)
            return mse
        except:
            return float('inf')

    def _acquisition_function(self, X):
        """Compute Expected Improvement acquisition function with uncertainty sampling"""
        if self.model_type == "gp":
            return self._compute_ei_gp(X)
        else:
            return self._compute_ei_linear(X)

    def _compute_ei_gp(self, X):
        """Compute EI for Gaussian Process model with enhanced exploration"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value
            y_best = np.min(self.y)
            
            # Compute EI with uncertainty sampling
            improvement = y_best - mean - self.epsilon  # Small epsilon to prevent division by zero
            z = improvement / (std + self.epsilon)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Add uncertainty term for exploration
            uncertainty_factor = 0.3 * std  # Reduced factor for better balance
            
            # Combine EI with uncertainty for better exploration-exploitation balance
            combined_acq = ei + uncertainty_factor
            
            return combined_acq.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ei_linear(self, X):
        """Compute EI for Linear model with enhanced exploration"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic (this is not true EI but maintains exploration)
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            # Add uncertainty term for better exploration
            uncertainty_factor = 0.3 * np.std(self.y)  # Reduced factor for efficiency
            
            return ei + uncertainty_factor
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _optimize_acquisition(self, X_init):
        """Optimize acquisition function using multiple strategies"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(1)
        
        best_acq_val = -np.inf
        best_point = None
        
        # Strategy 1: Local optimization from random starts with trust-region method (fewer restarts)
        for _ in range(min(3, max(1, self.acq_n_restarts // 2))):
            try:
                # Random start point
                x_start = X_init[self.rng.integers(len(X_init))].reshape(1, -1)
                
                # Use scipy minimize with trust-constr method for local optimization
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0][0],
                    x_start,
                    method='trust-constr',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'disp': False}
                )
                
                if result.success:
                    acq_val = self._acquisition_function(result.x.reshape(1, -1))[0][0]
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x.reshape(1, -1)
            except:
                continue
        
        # Strategy 2: Global optimization with differential evolution (fewer iterations)
        if best_point is None:
            try:
                # Use differential evolution for global optimization with fewer iterations
                bounds = list(zip(self.bounds[0], self.bounds[1]))
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0][0],
                    bounds,
                    seed=int(self.rng.integers(2**31 - 1)),
                    maxiter=20,  # Reduced iterations for efficiency
                    popsize=5    # Reduced population size
                )
                
                if result.success:
                    acq_val = self._acquisition_function(result.x.reshape(1, -1))[0][0]
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x.reshape(1, -1)
            except:
                pass
        
        # Strategy 3: Fallback to sample-based optimization
        if best_point is None:
            # Sample random points and pick the best
            sample_points = self._sample_points(10)  # Reduced number of samples
            acq_vals = self._acquisition_function(sample_points)
            max_idx = np.argmax(acq_vals.flatten())
            best_point = sample_points[max_idx].reshape(1, -1)
        
        return best_point

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Generate multiple starting points for optimization
        start_points = self._sample_points(min(20, max(5, self.acq_n_restarts * 3)))  # Reduced samples
        
        # Optimize acquisition function from these points
        next_point = self._optimize_acquisition(start_points)
        
        return next_point

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
                
            # Select next points for evaluation
            X_next = self._select_next_points(1)
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows (reduced frequency)
            if len(self.X) > 5 and self.n_evals < self.budget * 0.7 and self.n_evals % 3 == 0:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance (less frequent)
            if len(self.X) > 15 and self.n_evals < self.budget * 0.8 and self.n_evals % 5 == 0:
                try:
                    # Test if GP is significantly better than linear
                    test_indices = self.rng.choice(len(self.X), min(10, len(self.X)), replace=False)
                    X_test = self.X[test_indices]
                    y_test = self.y[test_indices]
                    
                    gp_error = self._compute_gp_performance(X_test, y_test)
                    lin_error = self._compute_linear_performance(X_test, y_test)
                    
                    # Update performance history
                    self.gp_performance_history.append(gp_error)
                    self.linear_performance_history.append(lin_error)
                    
                    # Keep only last 3 evaluations for model selection
                    if len(self.gp_performance_history) > 3:
                        self.gp_performance_history = self.gp_performance_history[-3:]
                        self.linear_performance_history = self.linear_performance_history[-3:]
                    
                    # Switch to linear if it's better (but not too early)
                    if len(self.gp_performance_history) >= 2:
                        gp_avg = np.mean(self.gp_performance_history[-2:])
                        lin_avg = np.mean(self.linear_performance_history[-2:])
                        
                        # Only switch if linear is significantly better
                        if lin_avg < gp_avg * 0.9 and self.model_type != "linear":
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                except:
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
