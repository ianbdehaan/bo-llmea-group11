from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
from scipy.optimize import differential_evolution
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class NovelBayesianOptimizationBO:
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
        self.acq_method = "hybrid"  # Can be "ei", "ucb", or "hybrid"
        
        # Budget management
        self.budget_phase = "initial"  # "initial", "exploration", "exploitation"

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
                    n_restarts_optimizer=min(3, max(1, self.n_evals // 15)),
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
        """Compute acquisition function based on selected method"""
        if self.model_type == "gp":
            if self.acq_method == "ei":
                return self._compute_ei_gp(X)
            elif self.acq_method == "ucb":
                return self._compute_ucb_gp(X)
            else:  # hybrid
                ei = self._compute_ei_gp(X)
                ucb = self._compute_ucb_gp(X)
                # Combine with a simple weighting scheme
                return 0.7 * ei + 0.3 * ucb
        else:
            if self.acq_method == "ei":
                return self._compute_ei_linear(X)
            elif self.acq_method == "ucb":
                return self._compute_ucb_linear(X)
            else:  # hybrid
                ei = self._compute_ei_linear(X)
                ucb = self._compute_ucb_linear(X)
                return 0.7 * ei + 0.3 * ucb

    def _compute_ei_gp(self, X):
        """Compute Expected Improvement for Gaussian Process model"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # Find best observed value
            y_best = np.min(self.y)
            
            # Compute EI
            improvement = y_best - mean - 1e-8  # Small epsilon to prevent division by zero
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            return ei.reshape(-1, 1)
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ucb_gp(self, X):
        """Compute Upper Confidence Bound for Gaussian Process model"""
        if self.gp_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Get predictions from GP
            mean, std = self.gp_model.predict(X, return_std=True)
            
            # UCB with exploration parameter kappa
            kappa = 2.0  # Fixed for simplicity, could be adaptive
            ucb = mean + kappa * std
            
            return -ucb.reshape(-1, 1)  # Negative because we maximize acquisition function
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ei_linear(self, X):
        """Compute Expected Improvement for Linear model"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic (this is not true EI but maintains exploration)
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            return ei
        except:
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_ucb_linear(self, X):
        """Compute Upper Confidence Bound for Linear model"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # Simple UCB using variance estimates
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple estimate of uncertainty (constant)
            uncertainty = np.ones_like(predictions) * 0.1
            kappa = 2.0
            
            ucb = predictions + kappa * uncertainty
            
            return -ucb.reshape(-1, 1)  # Negative because we maximize acquisition function
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
        restarts = min(10, max(3, self.acq_n_restarts // 2))
        for _ in range(restarts):
            # Sample initial points for optimization
            start_points = self._sample_points(5)
            
            try:
                # Use scipy's differential evolution to optimize acquisition function
                def objective(x):
                    x = np.array(x).reshape(1, -1)
                    acq_val = self._acquisition_function(x)[0]
                    return -float(acq_val)  # Negative because we minimize
                
                # Optimize with bounds
                bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
                
                result = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=20, 
                    popsize=5,
                    seed=int(self.rng.integers(2**31 - 1)),
                    strategy='best1bin',
                    atol=1e-6
                )
                
                if not result.success:
                    # Fallback to simple sampling if optimization fails
                    continue
                    
                # Check if this is better than current best
                acq_val = -result.fun
                if acq_val > best_acq_val:
                    best_acq_val = acq_val
                    best_point = result.x.reshape(1, -1)
            except:
                continue
        
        # If we didn't find a good point through optimization, use random sampling as fallback
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
            
            # Update model if we have sufficient data and budget allows
            if len(self.X) > 5 and self.n_evals < self.budget * 0.7:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except:
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance
            if len(self.X) > 15 and self.n_evals < self.budget * 0.8:
                try:
                    # Test if GP is significantly better than linear
                    # Use cross-validation to estimate performance
                    gp_pred = self.gp_model.predict(self.X, return_std=False)
                    if self.linear_model is not None:
                        lin_pred = self.linear_model.predict(self.scaler.transform(self.X)) if hasattr(self.linear_model, 'predict') else np.zeros(len(self.X))
                        
                        gp_error = np.mean((gp_pred.flatten() - self.y.flatten())**2)
                        lin_error = np.mean((lin_pred - self.y.flatten())**2) if len(lin_pred) == len(self.y) else 1e6
                        
                        # Switch to linear if it's significantly better (within 10% margin)
                        if lin_error < gp_error * 0.9 and self.model_type != "linear":
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                except:
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
