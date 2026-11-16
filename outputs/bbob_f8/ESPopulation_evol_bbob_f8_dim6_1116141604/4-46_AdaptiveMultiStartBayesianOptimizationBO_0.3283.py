from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdaptiveMultiStartBayesianOptimizationBO:
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
        self.n_restarts = 10  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_type = "gp"  # Current model type: "gp" or "linear"
        self.model_switch_threshold = 0.5  # Threshold to switch between models
        self.min_batch_size = 1
        self.max_batch_size = 5
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Fit either Gaussian Process or Linear model based on problem characteristics
        if self.model_type == "gp":
            return self._fit_gp_model(X, y)
        else:
            return self._fit_linear_model(X, y)
    
    def _fit_gp_model(self, X, y):
        # Fit Gaussian Process with Matern kernel
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
            n_restarts_optimizer=min(5, self.n_restarts),  # Reduced restarts for efficiency
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit linear model with standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use a small subset for linear regression to prevent overfitting
        max_samples = min(500, X.shape[0])
        if X.shape[0] > max_samples:
            indices = self.rng.choice(X.shape[0], max_samples, replace=False)
            X_subset = X_scaled[indices]
            y_subset = y[indices]
        else:
            X_subset = X_scaled
            y_subset = y
            
        model = LinearRegression()
        model.fit(X_subset, y_subset)
        
        # Store scaler for future predictions
        return (model, scaler)
    
    def _predict_model(self, model, X):
        # Predict using either GP or Linear model
        if isinstance(model, tuple):  # Linear model case
            model_obj, scaler = model
            X_scaled = scaler.transform(X)
            return model_obj.predict(X_scaled)
        else:  # GP model case
            return model.predict(X, return_std=False)
    
    def _predict_model_with_std(self, model, X):
        # Predict with standard deviation for GP models
        if isinstance(model, tuple):  # Linear model case - return constant std
            mu = self._predict_model(model, X)
            return mu, np.full(mu.shape, 1e-6)
        else:  # GP model case
            return model.predict(X, return_std=True)
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit appropriate model
        if self.model_type == "gp":
            model = self._fit_gp_model(self.X, self.y)
        else:
            model = self._fit_linear_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict_model_with_std(model, X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        exploitation_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.8)
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition_function(self, n_candidates=50):
        # Use a hybrid approach for acquisition function optimization
        # Start with multiple random initial points for better exploration
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        # Also try local search around top candidates for better refinement
        refined_candidates = []
        for i in range(min(self.batch_size, len(top_indices))):
            base_point = selected_points[i]
            
            # Add some random perturbation around the best points
            perturbation = self.rng.normal(0, 0.1, size=self.dim)
            perturbed_point = base_point + perturbation
            
            # Clip to bounds
            perturbed_point = np.clip(perturbed_point, self.bounds[0], self.bounds[1])
            
            refined_candidates.append(perturbed_point)
            
        if len(refined_candidates) > 0:
            refined_points = np.array(refined_candidates)
            acq_refined = self._acquisition_function(refined_points)
            best_refined_idx = np.argmax(acq_refined.flatten())
            selected_points[best_refined_idx] = refined_points[best_refined_idx]
        
        return selected_points
    
    def _select_next_points(self, batch_size):
        # Use hybrid strategy to optimize acquisition function
        n_candidates = max(50, batch_size * 20)
        
        # Select points using optimization approach
        selected_points = self._optimize_acquisition_function(n_candidates)
        
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
    
    def _switch_model(self):
        # Switch model based on problem characteristics
        if self.X is None or len(self.X) < 10:
            return False
            
        # Calculate how well we're fitting the data (simple metric)
        # Fit both models and compare performance
        gp_model = self._fit_gp_model(self.X, self.y)
        lin_model = self._fit_linear_model(self.X, self.y)
        
        # Simple performance estimate - for GP: get likelihood of current points
        try:
            gp_mu, gp_sigma = self._predict_model_with_std(gp_model, self.X)
            gp_log_likelihood = np.sum(norm.logpdf(self.y.flatten(), loc=gp_mu.flatten(), scale=gp_sigma.flatten()))
        except:
            gp_log_likelihood = -np.inf
            
        # For linear model: use MSE
        try:
            lin_pred = self._predict_model(lin_model, self.X)
            lin_mse = np.mean((self.y.flatten() - lin_pred.flatten()) ** 2)
        except:
            lin_mse = np.inf
            
        # Switch to linear if it's performing better or when we have many points
        if (gp_log_likelihood < -1000 or self.X.shape[0] > 50) and lin_mse < 1.0:
            if self.model_type != "linear":
                self.model_type = "linear"
                return True
        
        return False
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Check if we should switch model type
            if self.n_evals > 10 and self.n_evals % 20 == 0:
                self._switch_model()
            
            # Determine batch size based on remaining budget and current progress
            remaining_budget = self.budget - self.n_evals
            
            if remaining_budget <= 10:
                self.batch_size = 1
            elif remaining_budget <= 50:
                self.batch_size = min(3, max(self.min_batch_size, int(np.ceil(remaining_budget / 20))))
            else:
                # Dynamic batch size based on problem complexity and progress
                if self.X is not None and len(self.X) > 10:
                    batch_size_estimate = max(1, min(self.max_batch_size, 
                                                    int(np.sqrt(len(self.X)))))
                    self.batch_size = min(batch_size_estimate, remaining_budget)
                else:
                    self.batch_size = 1
            
            # Ensure we don't exceed the budget
            actual_batch_size = min(self.batch_size, remaining_budget)
            
            if actual_batch_size <= 0:
                break
                
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(actual_batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
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
