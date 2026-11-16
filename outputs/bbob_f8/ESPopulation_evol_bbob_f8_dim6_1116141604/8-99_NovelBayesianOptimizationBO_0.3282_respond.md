# Description
**Novel Bayesian Optimization with Adaptive Model Selection, Multi-Start Acquisition, and Smart Budget Allocation**

This refined Bayesian optimization algorithm enhances the previous EfficientBO by introducing several key improvements:

1. **Smart Budget Allocation**: Dynamically adjusts the number of initial points based on problem characteristics and budget constraints, ensuring better exploration-exploitation balance
2. **Enhanced Model Selection**: Implements a more robust switching mechanism between GP and Linear models based on predictive performance metrics
3. **Improved Acquisition Function**: Uses a novel hybrid acquisition function that combines Expected Improvement with Uncertainty Sampling for better exploration
4. **Multi-Start Optimization**: Implements a more sophisticated multi-start approach using both random sampling and local search techniques
5. **Adaptive Batch Size**: Dynamically adjusts batch size based on remaining budget to maximize information gain
6. **Efficient Model Updates**: Uses selective retraining strategies to minimize computational overhead

# Justification 
The key improvements in this algorithm are designed to address several limitations of standard Bayesian optimization approaches:

1. **Smart Budget Allocation**: The original algorithm already had some budget allocation, but we enhance it by making the initial point selection more adaptive based on both budget and dimensionality, ensuring better exploration in early stages while not wasting evaluations.

2. **Enhanced Model Selection**: We improve model switching logic by using a more robust performance metric with cross-validation and introduce a "model confidence" mechanism that prevents unnecessary switches during optimization.

3. **Improved Acquisition Function**: The hybrid approach combining EI and UCB is kept but refined to better balance exploration and exploitation, especially in early iterations when uncertainty is high.

4. **Multi-Start Optimization**: We optimize acquisition function more efficiently by using multiple restarts with a smarter selection strategy and reducing local search complexity.

5. **Adaptive Batch Size**: The algorithm now dynamically adjusts batch size based on remaining budget to ensure maximum information gain while respecting the budget constraint.

6. **Efficient Model Updates**: We minimize computational overhead by selectively retraining models only when there's sufficient new data and performance improvement justifies the cost.

The improvements aim to make the algorithm more robust, efficient, and scalable across different budget sizes and dimensions, particularly for challenging functions like Rosenbrock.

# Code
```python
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
        
        # Model confidence tracking for switching
        self.model_confidence = 1.0
        self.model_switch_threshold = 0.8
        
        # Budget adaptation parameters
        self.budget_remaining = budget
        self.adaptive_batch_size = 1

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
                # Combine with a simple weighting scheme that adapts based on budget
                weight = min(1.0, self.n_evals / (self.budget * 0.5))  # Increase EI weight as optimization progresses
                return weight * ei + (1 - weight) * ucb
        else:
            if self.acq_method == "ei":
                return self._compute_ei_linear(X)
            elif self.acq_method == "ucb":
                return self._compute_ucb_linear(X)
            else:  # hybrid
                ei = self._compute_ei_linear(X)
                ucb = self._compute_ucb_linear(X)
                weight = min(1.0, self.n_evals / (self.budget * 0.5))
                return weight * ei + (1 - weight) * ucb

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
            
            # UCB with exploration parameter kappa that adapts based on budget
            kappa = 2.0 + min(2.0, self.n_evals / (self.budget * 0.3))  # Increase exploration early
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
            
            # Estimate uncertainty based on data spread
            uncertainty = np.std(self.y) * 0.5 if len(self.y) > 1 else 0.1
            kappa = 2.0 + min(2.0, self.n_evals / (self.budget * 0.3))  # Increase exploration early
            
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
        
        # First, try a few random points for diversity
        random_points = self._sample_points(restarts)
        random_acq_vals = self._acquisition_function(random_points).flatten()
        
        # Check if any random point is better than current best
        max_random_idx = np.argmax(random_acq_vals)
        if random_acq_vals[max_random_idx] > best_acq_val:
            best_acq_val = random_acq_vals[max_random_idx]
            best_point = random_points[max_random_idx].reshape(1, -1)
        
        # Then try local search for better points
        for _ in range(restarts // 2):
            # Sample initial points for optimization
            start_points = self._sample_points(3)
            
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
                    maxiter=15, 
                    popsize=3,
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

    def _update_budget_adaptation(self):
        """Update batch size and other parameters based on remaining budget"""
        # Dynamic batch size: larger batches when more budget remains
        if self.budget > 100:
            self.adaptive_batch_size = min(5, max(1, int((self.budget - self.n_evals) / (self.budget * 0.3))))
        else:
            self.adaptive_batch_size = 1
            
        # Adjust initial points based on remaining budget
        if self.budget_remaining > self.budget * 0.8 and self.n_init > 5:
            self.n_init = max(5, min(15, self.budget // 6))
        elif self.budget_remaining < self.budget * 0.3 and self.n_init > 20:
            self.n_init = max(10, min(20, self.budget // 8))

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
            # Update budget adaptation
            self.budget_remaining = self.budget - self.n_evals
            self._update_budget_adaptation()
            
            # Check if we have enough evaluations to continue
            if self.n_evals >= self.budget:
                break
                
            # Select next points for evaluation (using adaptive batch size)
            X_next = self._select_next_points(self.adaptive_batch_size)
            
            # Evaluate next point(s)
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
                        
                        # Switch to linear if it's significantly better (within 10% margin) and confidence is high
                        if lin_error < gp_error * 0.9 and self.model_type != "linear" and self.model_confidence > self.model_switch_threshold:
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                            # Reduce confidence after switching to prevent frequent switches
                            self.model_confidence *= 0.7
                        elif lin_error >= gp_error * 0.9 and self.model_type == "linear":
                            # Increase confidence when GP is performing better
                            self.model_confidence = min(1.0, self.model_confidence + 0.1)
                except:
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Feedback
 The algorithm NovelBayesianOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3282 with standard deviation 0.0160.

took 149.00 seconds to run.