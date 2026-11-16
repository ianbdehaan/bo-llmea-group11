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

class AdaptiveMultiStartBO:
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
        self.acq_func_type = "hybrid"  # Hybrid acquisition function
        self.n_restarts = min(20, max(5, int(np.ceil(self.dim * 2))))  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_type = "gp"  # Initially use Gaussian Process
        self.model_update_freq = max(1, int(np.ceil(np.sqrt(self.budget))))
        self.use_local_search = True
        self.adaptive_factor = 0.5  # Controls exploration-exploitation balance
        
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
                n_restarts_optimizer=min(10, self.n_restarts),
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            gp.fit(X_subset, y_subset)
            return gp
        else:
            # Fit Linear model with standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            return (lr, scaler)
    
    def _predict_model(self, model, X):
        # Predict using the appropriate model
        if isinstance(model, tuple):  # Linear model
            lr, scaler = model
            X_scaled = scaler.transform(X)
            mu = lr.predict(X_scaled)
            sigma = np.full(mu.shape, 1e-6)  # Simple uncertainty assumption for linear model
            return mu, sigma
        else:  # Gaussian Process model
            return model.predict(X, return_std=True)
    
    def _acquisition_function(self, X):
        # Hybrid acquisition function combining Expected Improvement and Uncertainty Sampling
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Fit GP model
        model = self._fit_model(self.X, self.y)
        
        # Predict mean and std at candidate points
        mu, sigma = self._predict_model(model, X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Uncertainty Sampling (high uncertainty is good for exploration)
        us = -sigma
        
        # Hybrid acquisition function
        hybrid_acq = self.adaptive_factor * ei + (1 - self.adaptive_factor) * us
        
        return hybrid_acq.reshape(-1, 1)
    
    def _optimize_acquisition(self, n_candidates=50):
        # Multi-start optimization of acquisition function
        # Start with random sampling for global exploration
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        # If local search is enabled and we have enough points, try local optimization
        if self.use_local_search and len(selected_points) > 0:
            # Try to refine the best candidates using a local search approach
            refined_candidates = []
            for point in selected_points[:min(5, len(selected_points))]:  # Refine top 5 only
                # Perturb the point and evaluate nearby points
                perturbed_points = []
                for _ in range(10):
                    # Add small random perturbation
                    perturbation = self.rng.normal(0, 0.1, self.dim) * (self.bounds[1] - self.bounds[0])
                    perturbed_point = np.clip(point + perturbation, self.bounds[0], self.bounds[1])
                    perturbed_points.append(perturbed_point)
                
                # Evaluate these points
                if len(perturbed_points) > 0:
                    perturbed_array = np.array(perturbed_points)
                    acq_values_perturbed = self._acquisition_function(perturbed_array)
                    best_idx = np.argmax(acq_values_perturbed.flatten())
                    refined_candidates.append(perturbed_array[best_idx])
                else:
                    refined_candidates.append(point)
            
            # If we found better points, use them
            if len(refined_candidates) > 0:
                refined_array = np.array(refined_candidates)
                acq_values_refined = self._acquisition_function(refined_array)
                best_refined_idx = np.argmax(acq_values_refined.flatten())
                selected_points = np.array([refined_array[best_refined_idx]])
        
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
        # Switch model based on performance metrics
        if self.X is None or self.X.shape[0] < 10:
            return
            
        # Simple heuristic: switch to linear if we have sufficient data and GP performance degrades
        try:
            gp = self._fit_model(self.X, self.y)
            lr = LinearRegression()
            
            # Compare predictive performance on a subset of data
            n_samples = min(100, self.X.shape[0])
            indices = self.rng.choice(self.X.shape[0], n_samples, replace=False)
            X_test = self.X[indices]
            y_test = self.y[indices]
            
            # GP prediction
            gp_mu, _ = self._predict_model(gp, X_test)
            gp_rmse = np.sqrt(np.mean((gp_mu.flatten() - y_test.flatten()) ** 2))
            
            # Linear regression prediction  
            lr.fit(self.X, self.y)
            lr_pred = lr.predict(X_test)
            lr_rmse = np.sqrt(np.mean((lr_pred.flatten() - y_test.flatten()) ** 2))
            
            # Switch to linear if it performs better
            if lr_rmse < gp_rmse * 0.9:  # Linear is significantly better
                self.model_type = "linear"
            elif gp_rmse < lr_rmse * 0.9:  # GP is significantly better
                self.model_type = "gp"
        except:
            pass
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Adaptive batch size adjustment based on remaining budget
        if self.budget > 200:
            self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
        else:
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Update model type based on performance
                if self.n_evals % self.model_update_freq == 0:
                    self._switch_model()
                
                # Select next batch of points to evaluate
                next_batch = self._optimize_acquisition(max(20, self.batch_size * 10))
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Update batch size based on remaining budget and progress
                if self.budget > 200:
                    self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
                    
                # Adjust exploration-exploitation balance as optimization progresses
                self.adaptive_factor = max(0.1, 0.5 - (self.n_evals / self.budget) * 0.4)
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
