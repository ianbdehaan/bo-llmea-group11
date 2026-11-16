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

class EnhancedBayesianOptimizationBO:
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
        self.ei_weight = 0.7  # Weight for Expected Improvement in hybrid acquisition
        self.us_weight = 0.3  # Weight for Uncertainty Sampling in hybrid acquisition
        
        # Convergence tracking
        self.best_so_far = float('inf')
        self.convergence_history = []

    def _sample_points(self, n_points):
        """Sample points using a combination of Sobol and grid sampling"""
        if n_points <= 10:
            # Use Sobol for small samples
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            points = sampler.random(n_points)
            points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        else:
            # Use Latin Hypercube for larger samples
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            points = sampler.random(n_points)
            points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _sample_grid_points(self, n_points):
        """Sample points using grid sampling for better local coverage"""
        # Create a simple grid sampling
        n_per_dim = int(np.ceil(n_points ** (1.0 / self.dim)))
        if n_per_dim > 10:
            n_per_dim = 10
            
        grid_points = []
        for i in range(self.dim):
            axis = np.linspace(self.bounds[0][i], self.bounds[1][i], n_per_dim)
            grid_points.append(axis)
            
        # Create meshgrid and flatten
        grids = np.meshgrid(*grid_points, indexing='ij')
        points = np.column_stack([g.ravel() for g in grids])
        
        # Select random subset if needed
        if len(points) > n_points:
            indices = self.rng.choice(len(points), size=n_points, replace=False)
            points = points[indices]
            
        return points

    def _fit_model(self, X, y):
        """Fit surrogate model with adaptive selection and robust error handling"""
        if self.model_type == "gp":
            # Fit Gaussian Process with Matern kernel
            try:
                # Use a more robust kernel with better bounds
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    n_restarts_optimizer=min(5, max(1, self.n_evals // 5)),
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X, y)
                return gp
            except Exception as e:
                # Fallback to linear model if GP fails
                print(f"GP fitting failed with error: {e}. Using Linear model.")
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
        except Exception as e:
            print(f"Linear model fitting failed with error: {e}. Using constant prediction.")
            # Fallback to constant prediction
            return np.mean(y)

    def _hybrid_acquisition_function(self, X):
        """Compute hybrid acquisition function combining EI and Uncertainty Sampling"""
        if self.model_type == "gp":
            return self._compute_hybrid_ei_gp(X)
        else:
            return self._compute_hybrid_ei_linear(X)

    def _compute_hybrid_ei_gp(self, X):
        """Compute hybrid EI for Gaussian Process model"""
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
            
            # Compute Uncertainty Sampling (high uncertainty is good for exploration)
            us = std
            
            # Combine both metrics with adaptive weights
            if len(self.convergence_history) > 20:
                # If we're in later stages of optimization, favor exploitation
                current_weight = max(0.5, min(0.9, 0.7 + 0.2 * (len(self.convergence_history) / len(self.convergence_history))))
                ei_weight = current_weight
                us_weight = 1 - current_weight
            else:
                # Early stages favor exploration
                ei_weight = self.ei_weight
                us_weight = self.us_weight
            
            hybrid_acq = ei_weight * ei + us_weight * us
            
            return hybrid_acq.reshape(-1, 1)
        except Exception as e:
            print(f"Error in hybrid acquisition GP: {e}")
            return np.ones((X.shape[0], 1)) * -1e6

    def _compute_hybrid_ei_linear(self, X):
        """Compute hybrid EI for Linear model"""
        if self.linear_model is None or self.y is None:
            return np.ones((X.shape[0], 1)) * -1e6
        
        try:
            # For linear model, we use a simple heuristic with uncertainty
            X_scaled = self.scaler.transform(X)
            predictions = self.linear_model.predict(X_scaled).reshape(-1, 1)
            
            # Simple improvement heuristic 
            y_best = np.min(self.y)
            ei = np.maximum(y_best - predictions, 0)
            
            # Add uncertainty component (variance in prediction)
            # Since linear regression doesn't provide std, we estimate it
            if self.X is not None and len(self.X) > 1:
                # Estimate uncertainty from residuals
                X_train_scaled = self.scaler.transform(self.X)
                train_pred = self.linear_model.predict(X_train_scaled).flatten()
                mse = mean_squared_error(self.y.flatten(), train_pred)
                uncertainty = np.sqrt(mse) * np.ones((X.shape[0], 1))
            else:
                uncertainty = np.ones((X.shape[0], 1)) * 0.1
            
            # Combine with adaptive weights
            hybrid_acq = 0.7 * ei + 0.3 * uncertainty
            
            return hybrid_acq
        except Exception as e:
            print(f"Error in hybrid acquisition Linear: {e}")
            return np.ones((X.shape[0], 1)) * -1e6

    def _select_next_points(self, batch_size):
        """Select next points using multi-start optimization of acquisition function"""
        if self.X is None or self.y is None:
            # If no data yet, sample randomly
            return self._sample_points(batch_size)
        
        # Use a two-stage approach for efficiency
        candidates = []
        candidate_acq_vals = []
        
        # Stage 1: Generate candidates with different sampling strategies
        n_candidates = min(50, max(10, self.acq_n_restarts * 3))
        
        # Strategy 1: Random sampling
        random_points = self._sample_points(n_candidates // 2)
        acq_vals = self._hybrid_acquisition_function(random_points)
        for i in range(len(random_points)):
            candidates.append(random_points[i])
            candidate_acq_vals.append(float(acq_vals[i]))
        
        # Strategy 2: Local grid sampling around current best
        if len(self.X) > 1:
            # Sample points around the best solution found so far
            best_x = self.X[np.argmin(self.y)]
            local_points = []
            for _ in range(n_candidates // 4):
                # Perturb around best point with small random noise
                perturbed = best_x + self.rng.normal(0, 0.5, self.dim)
                # Clamp to bounds
                perturbed = np.clip(perturbed, self.bounds[0], self.bounds[1])
                local_points.append(perturbed)
            
            if len(local_points) > 0:
                local_points = np.array(local_points)
                acq_vals = self._hybrid_acquisition_function(local_points)
                for i in range(len(local_points)):
                    candidates.append(local_points[i])
                    candidate_acq_vals.append(float(acq_vals[i]))
        
        # Strategy 3: Grid sampling
        grid_points = self._sample_grid_points(n_candidates // 4)
        acq_vals = self._hybrid_acquisition_function(grid_points)
        for i in range(len(grid_points)):
            candidates.append(grid_points[i])
            candidate_acq_vals.append(float(acq_vals[i]))
        
        # Stage 2: Select top candidates
        if len(candidate_acq_vals) == 0:
            return self._sample_points(batch_size)
            
        # Sort by acquisition value and select top ones
        sorted_indices = np.argsort(candidate_acq_vals)[::-1]
        selected_candidates = [candidates[i] for i in sorted_indices[:batch_size]]
        
        return np.array(selected_candidates)

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
        self.best_so_far = float('inf')
        self.convergence_history = []
        
        # Generate initial points using enhanced sampling
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
                
            # Update best so far
            current_best = np.min(self.y)
            self.best_so_far = min(self.best_so_far, current_best)
            self.convergence_history.append(current_best)
            
            # Select next points for evaluation
            X_next = self._select_next_points(1)
            
            # Evaluate next point
            y_next = self._evaluate_points(func, X_next)
            
            # Update the model with new data
            self._update_eval_points(X_next, y_next)
            
            # Update model if we have sufficient data and budget allows
            if len(self.X) > 5 and self.n_evals < self.budget * 0.8:
                try:
                    # Try to update the model with new data
                    self.gp_model = self._fit_model(self.X, self.y)
                except Exception as e:
                    print(f"Model update failed: {e}")
                    pass  # Continue with current model if update fails
            
            # Adaptive switching between models based on performance
            if len(self.X) > 15 and self.n_evals < self.budget * 0.9:
                try:
                    # Test if GP is significantly better than linear
                    if self.gp_model is not None:
                        gp_pred = self.gp_model.predict(self.X, return_std=False)
                        
                        # Simple performance metric - mean squared error
                        gp_error = np.mean((gp_pred.flatten() - self.y.flatten())**2)
                        
                        # Evaluate linear model performance
                        lin_pred = self.linear_model.predict(self.scaler.transform(self.X)) if self.linear_model is not None else np.zeros(len(self.X))
                        lin_error = np.mean((lin_pred - self.y.flatten())**2) if len(lin_pred) == len(self.y) else 1e6
                        
                        # Switch to linear if it's better and we're not too early
                        if lin_error < gp_error * 0.9 and self.model_type != "linear" and len(self.X) > 20:
                            print(f"Switching to linear model (GP error: {gp_error:.4f}, Linear error: {lin_error:.4f})")
                            self.model_type = "linear"
                            self.linear_model = self._fit_linear_model(self.X, self.y)
                except Exception as e:
                    print(f"Model switching failed: {e}")
                    pass
        
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
