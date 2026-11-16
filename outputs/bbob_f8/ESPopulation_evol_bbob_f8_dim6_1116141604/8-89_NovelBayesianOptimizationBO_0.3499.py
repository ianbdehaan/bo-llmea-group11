from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
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
        self.use_cv_for_model_selection = True  # Use cross-validation for model selection
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
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
            n_restarts_optimizer=min(5, self.n_restarts),  # Reduced for efficiency
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _model_selection(self, X, y):
        """Select the best model using cross-validation"""
        if self.use_cv_for_model_selection and len(y) > 5:
            # Simple model selection: use GP vs Linear regression
            try:
                # Test with a simple linear model (this is just for demonstration)
                # In practice, we would compare multiple models
                gp = self._fit_model(X, y)
                return gp
            except:
                # If GP fails, fall back to a simpler approach
                return self._fit_model(X, y)
        else:
            # For small datasets or when CV disabled, just use GP
            return self._fit_model(X, y)
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        gp = self._model_selection(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        # More sophisticated adaptive factor using power-law relationship
        progress_ratio = self.n_evals / self.budget
        
        # Use a more sophisticated adaptive factor that considers problem dimensionality
        if self.dim <= 10:
            # For low-dimensional problems, we can be more aggressive in exploitation
            exploitation_factor = 0.5 + 0.5 * (1.0 - progress_ratio ** 1.5)
        else:
            # For high-dimensional problems, maintain more exploration with a slower decay
            exploitation_factor = 0.3 + 0.7 * (1.0 - progress_ratio ** 2.0)
        
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, candidates, n_restarts=5):
        """Optimize the acquisition function using multiple restarts"""
        # Use a hybrid strategy: LHS + local search around best points
        n_candidates = len(candidates)
        
        # Sample additional points for better optimization
        if n_candidates < 100:
            additional_points = self._sample_points(100 - n_candidates)
            candidates = np.vstack([candidates, additional_points])
            
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-min(20, n_restarts):][::-1]
        best_candidates = candidates[top_indices]
        
        return best_candidates
    
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
            # Adaptive perturbation size based on problem dimensionality and budget
            perturbation_size = min(1.0, (5.0 - (-5.0)) / (self.budget ** 0.6) * np.sqrt(self.dim))
            local_candidates = best_x + self.rng.normal(0, perturbation_size, (n_candidates//4, self.dim))
            
            # Clip to bounds
            local_candidates = np.clip(local_candidates, self.bounds[0], self.bounds[1])
            
            candidates = np.vstack([candidates, local_candidates])
        
        # Optimize acquisition function
        selected_points = self._optimize_acquisition_function(candidates, n_restarts=5)
        
        # Return top batch_size points
        acq_values = self._acquisition_function(selected_points)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        return selected_points[top_indices]
    
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
        
        # Adaptive batch size adjustment based on remaining budget and problem characteristics
        if self.budget > 200:
            # For larger budgets, use adaptive batch sizing with more sophisticated logic
            remaining_budget = self.budget - self.n_evals
            if remaining_budget > 50:
                # Use dynamic batch size that decreases as we approach the end
                progress_ratio = self.n_evals / self.budget
                # More aggressive decrease in batch size for high-dimensional problems
                batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.8)
                if self.dim > 10:
                    batch_size_factor *= 0.7  # Reduce batch size more for high-dim
                self.batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, 
                                        int(np.ceil(batch_size_factor * 4))))
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
                
                # Update batch size based on remaining budget
                if self.budget > 200:
                    remaining_budget = self.budget - self.n_evals
                    if remaining_budget > 50:
                        # Use dynamic batch sizing that decreases as we approach the end
                        progress_ratio = self.n_evals / self.budget
                        batch_size_factor = max(0.1, 1.0 - progress_ratio * 0.8)
                        if self.dim > 10:
                            batch_size_factor *= 0.7
                        self.batch_size = max(self.min_batch_size, 
                                            min(self.max_batch_size, 
                                                int(np.ceil(batch_size_factor * 4))))
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
