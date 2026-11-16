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

class AdaptiveEnsembleBO:
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
        self.n_restarts = 3  # Reduced number of restarts for efficiency
        self.batch_size = 1  # Initial batch size
        self.model_switch_threshold = 0.1  # Threshold to switch between models
        
        # Model storage
        self.gp_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        
        # Track model performance
        self.gp_performance = []
        self.linear_performance = []
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
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
            n_restarts_optimizer=self.n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_subset, y_subset)
        return gp
    
    def _fit_linear_model(self, X, y):
        # Fit Linear model
        X_scaled = self.scaler.fit_transform(X)
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        return lr
    
    def _predict_gp(self, X):
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
        
        try:
            mu, sigma = self.gp_model.predict(X, return_std=True)
            return mu.reshape(-1, 1), sigma.reshape(-1, 1)
        except:
            # Fallback to constant prediction if GP fails
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
    
    def _predict_linear(self, X):
        if self.linear_model is None:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1)) * 0.1
            
        try:
            X_scaled = self.scaler.transform(X)
            mu = self.linear_model.predict(X_scaled).reshape(-1, 1)
            # Simple uncertainty estimate
            sigma = np.ones((X.shape[0], 1)) * 0.1
            return mu, sigma
        except:
            # Fallback to constant prediction if Linear model fails
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1)) * 0.1
    
    def _predict_ensemble(self, X):
        # Predict using both models and combine based on performance
        gp_mu, gp_sigma = self._predict_gp(X)
        linear_mu, linear_sigma = self._predict_linear(X)
        
        # Simple ensemble: weight by inverse of uncertainty (more uncertain = less weight)
        # This is a simplified approach - in practice, we could use more sophisticated weighting
        if len(self.gp_performance) > 0 and len(self.linear_performance) > 0:
            gp_score = np.mean(self.gp_performance)
            linear_score = np.mean(self.linear_performance)
            
            # Weight based on relative performance (lower is better)
            total_score = gp_score + linear_score
            if total_score > 0:
                gp_weight = linear_score / total_score
                linear_weight = gp_score / total_score
            else:
                gp_weight = 0.5
                linear_weight = 0.5
        else:
            gp_weight = 0.5
            linear_weight = 0.5
            
        mu = gp_weight * gp_mu + linear_weight * linear_mu
        sigma = gp_weight * gp_sigma + linear_weight * linear_sigma
        
        return mu, sigma
    
    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration
        if self.y is None or self.X is None:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Predict using ensemble
        mu, sigma = self._predict_ensemble(X)
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        if self.n_evals < self.budget * 0.2:
            # Early phase: more exploration
            exploitation_factor = 0.7
        elif self.n_evals < self.budget * 0.5:
            # Middle phase: balanced
            exploitation_factor = 0.9
        else:
            # Late phase: more exploitation
            exploitation_factor = 1.2
            
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _optimize_acquisition(self, n_candidates):
        # Hybrid optimization of acquisition function
        # Start with multiple random initial points for better exploration
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:][::-1]
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
    
    def _update_model_performance(self, X_test, y_test):
        # Update performance metrics for models
        if self.gp_model is not None:
            try:
                gp_pred, _ = self._predict_gp(X_test)
                gp_mse = np.mean((gp_pred - y_test)**2)
                self.gp_performance.append(gp_mse)
                if len(self.gp_performance) > 5:
                    self.gp_performance.pop(0)
            except:
                pass
                
        if self.linear_model is not None:
            try:
                linear_pred, _ = self._predict_linear(X_test)
                linear_mse = np.mean((linear_pred - y_test)**2)
                self.linear_performance.append(linear_mse)
                if len(self.linear_performance) > 5:
                    self.linear_performance.pop(0)
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
        
        # Fit initial models
        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.linear_model = self._fit_linear_model(self.X, self.y)
        
        # Adaptive batch size adjustment based on remaining budget and iteration progress
        # Start with a larger batch if we have more budget
        if self.budget > 500:
            self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 100))))
        elif self.budget > 200:
            self.batch_size = min(3, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
        else:
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1:
                # Select next batch of points to evaluate
                next_batch = self._optimize_acquisition(max(50, self.batch_size * 30))
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Retrain models on all data
                self.gp_model = self._fit_gp_model(self.X, self.y)
                self.linear_model = self._fit_linear_model(self.X, self.y)
                
                # Update model performance (using a subset of recent data for efficiency)
                if self.X.shape[0] > 5:
                    test_indices = self.rng.choice(self.X.shape[0], min(5, self.X.shape[0]), replace=False)
                    X_test = self.X[test_indices]
                    y_test = self.y[test_indices]
                    self._update_model_performance(X_test, y_test)
                
                # Update batch size based on remaining budget and progress
                if self.budget > 500:
                    self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 100))))
                elif self.budget > 200:
                    self.batch_size = min(3, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
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
