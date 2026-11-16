from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import Ridge
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
        self.n_init = max(10, min(20, int(np.ceil(np.sqrt(self.dim) * 5))))
        
        # For reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.acq_func_type = "ei"  # Expected Improvement
        self.n_restarts = 5  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.model_update_threshold = 20  # Minimum points before updating model
        self.model_selection_threshold = 50  # Number of points after which to consider switching models
        
        # Model selection parameters
        self.current_model = "gp"  # "gp" or "linear"
        self.gp_model = None
        self.linear_model = None
        self.model_performance_history = []
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_gp_model(self, X, y):
        # Fit Gaussian Process with Matern kernel
        # Use a small number of restarts for efficiency
        kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
        
        # Use a subset of data points if too many to avoid numerical issues
        max_samples = min(1000, X.shape[0])
        if X.shape[0] > max_samples:
            indices = self.rng.choice(X.shape[0], max_samples, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset = X
            y_subset = y
            
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=min(5, self.n_restarts),
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit Linear model (Ridge regression) for efficiency
        ridge = Ridge(alpha=1.0, random_state=int(self.rng.integers(2**31 - 1)))
        ridge.fit(X, y)
        return ridge
    
    def _predict_gp(self, gp, X):
        # Predict mean and std at candidate points using GP
        mu, sigma = gp.predict(X, return_std=True)
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        return mu, sigma
    
    def _predict_linear(self, model, X):
        # Predict mean using linear model
        mu = model.predict(X)
        # For linear model, we can estimate uncertainty via cross-validation
        # For simplicity in this implementation, return constant uncertainty
        sigma = np.full(mu.shape, 1e-4)  # Small fixed uncertainty
        return mu, sigma
    
    def _predict(self, X):
        # Predict using the current model
        if self.current_model == "gp" and self.gp_model is not None:
            return self._predict_gp(self.gp_model, X)
        else:
            return self._predict_linear(self.linear_model, X)
    
    def _calculate_model_performance(self, model_type):
        # Simple cross-validation for model performance estimation
        if self.X is None or len(self.X) < 5:
            return 0.0
            
        # Use a small subset for CV to save computation time
        n_samples = min(100, self.X.shape[0])
        indices = self.rng.choice(self.X.shape[0], n_samples, replace=False)
        X_subset = self.X[indices]
        y_subset = self.y[indices].flatten()
        
        if model_type == "gp":
            # Estimate GP performance using leave-one-out CV
            try:
                gp = self._fit_gp_model(X_subset, y_subset.reshape(-1, 1))
                # Simple estimate of log-likelihood or prediction error
                mu, _ = self._predict_gp(gp, X_subset)
                mse = np.mean((y_subset - mu)**2)
                return -mse  # Negative MSE as performance metric (higher is better)
            except:
                return -np.inf
        else:
            # Estimate linear model performance
            try:
                linear_model = self._fit_linear_model(X_subset, y_subset)
                # Use cross-validation for robustness
                scores = cross_val_score(linear_model, X_subset, y_subset, cv=min(5, len(X_subset)), 
                                       scoring='neg_mean_squared_error')
                return np.mean(scores)  # Negative MSE as performance metric (higher is better)
            except:
                return -np.inf
    
    def _select_best_model(self):
        # Select the best model based on recent performance
        if self.X is None or len(self.X) < 10:
            return "gp"  # Default to GP for small datasets
            
        gp_performance = self._calculate_model_performance("gp")
        linear_performance = self._calculate_model_performance("linear")
        
        # If both models have reasonable performance, prefer the one with higher score
        if gp_performance > linear_performance + 0.1:  # Prefer GP when significantly better
            return "gp"
        elif linear_performance > gp_performance + 0.1:  # Prefer Linear when significantly better
            return "linear"
        else:
            # When performance is similar, prefer GP for its flexibility (except early in optimization)
            if len(self.X) < self.model_selection_threshold:
                return "gp"  # Early on prefer GP for exploration
            else:
                return "gp"  # Default to GP for better performance
    
    def _update_model(self):
        # Update the model based on current data and performance metrics
        if self.X is None or len(self.X) < 2:
            return
            
        # Select best model type
        new_model = self._select_best_model()
        
        # Only update if model selection changed or we have enough points for reliable updates
        if (new_model != self.current_model and len(self.X) >= 10) or \
           (len(self.X) >= self.model_update_threshold):
            
            self.current_model = new_model
            
            if self.current_model == "gp":
                self.gp_model = self._fit_gp_model(self.X, self.y)
                # Reset linear model to save memory
                self.linear_model = None
            else:
                self.linear_model = self._fit_linear_model(self.X, self.y.flatten())
                # Reset GP model to save memory
                self.gp_model = None
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration and uncertainty sampling
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict(X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add uncertainty sampling component to encourage exploration in early iterations
        # This helps avoid getting stuck in local minima
        if len(self.X) < self.n_init * 0.5:  # Early iterations - more exploration
            uncertainty_factor = 1.0 + 2.0 * np.exp(-len(self.X) / (self.n_init * 0.3))
        else:
            uncertainty_factor = 1.0
            
        ei *= uncertainty_factor
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        progress_ratio = self.n_evals / self.budget
        
        # More sophisticated adaptive factor - use different strategies based on problem dimensionality
        if self.dim <= 10:
            # For low-dimensional problems, we can be more aggressive in exploitation
            exploitation_factor = max(0.2, 1.0 - progress_ratio * 0.7)
        else:
            # For high-dimensional problems, maintain more exploration
            exploitation_factor = max(0.1, 1.0 - progress_ratio * 0.9)
        
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(50, batch_size * 30)
        
        # Sample candidates using LHS
        candidates = self._sample_points(n_candidates)
        
        # Add some local search points around the current best
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Generate perturbations around the best point
            perturbation_size = min(1.0, 2.0 * (5.0 - (-5.0)) / self.budget)  # Dynamic perturbation size
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y_values = []
        for x in X:
            y_values.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)
    
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
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Update the model with initial points
        self._update_model()
        
        # Adaptive batch size adjustment based on remaining budget and problem characteristics
        if self.budget > 200:
            # For larger budgets, use adaptive batch sizing with more sophisticated logic
            remaining_budget = self.budget - self.n_evals
            if remaining_budget > 50:
                # Use dynamic batch size that decreases as we approach the end
                batch_size_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.7)
                self.batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, 
                                        int(np.ceil(batch_size_factor * 3))))
            else:
                # Use small batch size for final evaluations
                self.batch_size = 1
        else:
            # For smaller budgets, stick to single evaluation
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Update model periodically
                if self.n_evals % max(10, int(np.ceil(self.budget / 20))) == 0:
                    self._update_model()
                
                # Update batch size based on remaining budget
                if self.budget > 200:
                    remaining_budget = self.budget - self.n_evals
                    if remaining_budget > 50:
                        # Use dynamic batch sizing that decreases as we approach the end
                        batch_size_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.7)
                        self.batch_size = max(self.min_batch_size, 
                                            min(self.max_batch_size, 
                                                int(np.ceil(batch_size_factor * 3))))
                    else:
                        # Use small batch size for final evaluations
                        self.batch_size = 1
            else:
                # If we don't have enough points, evaluate more initial points
                remaining_budget = self.budget - self.n_evals
                if remaining_budget > 0:
                    additional_points = self._sample_points(min(remaining_budget, 10))
                    additional_values = self._evaluate_points(func, additional_points)
                    self._update_eval_points(additional_points, additional_values)
        
        # Return best point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
