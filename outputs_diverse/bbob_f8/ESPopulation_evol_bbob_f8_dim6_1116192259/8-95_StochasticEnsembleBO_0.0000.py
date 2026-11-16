from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution
from scipy.stats import norm
import warnings

class StochasticEnsembleBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Ensemble of models
        self.models = []
        self.model_weights = []
        self.model_performance = []
        
        # Initialize models
        self._initialize_models()
        
        # Acquisition function parameters
        self.acq_func_type = 0  # 0: EI, 1: PI, 2: Thompson Sampling
        self.acq_func_history = []
        self.acq_func_weights = [0.33, 0.33, 0.34]  # weights for the three acquisition functions
        
        # Exploration parameters
        self.exploration_weight = 0.1
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # For batch selection
        self.batch_size = 1
        
        # Model selection parameters
        self.model_selection_window = 5
        self.model_selection_threshold = 0.1
        
        # Performance tracking
        self.best_y_history = []

    def _initialize_models(self):
        """Initialize ensemble of surrogate models"""
        # Gaussian Process with ARD Matern kernel
        gp_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        gp_kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        gp_model = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        
        # Extra Trees
        et_model = ExtraTreesRegressor(
            n_estimators=10,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        
        self.models = [gp_model, rf_model, et_model]
        self.model_weights = [1.0/len(self.models)] * len(self.models)  # Equal initial weights
        self.model_performance = [0.0] * len(self.models)

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling"""
        # Create Sobol sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_models(self, X, y):
        """Fit all models in the ensemble"""
        for i, model in enumerate(self.models):
            try:
                if isinstance(model, GaussianProcessRegressor):
                    # For GP, we need to fit with the current data
                    model.fit(X, y)
                else:
                    # For tree-based models
                    model.fit(X, y.ravel())
            except Exception as e:
                # If fitting fails, keep the model as is or reset weights
                warnings.warn(f"Model fitting failed: {e}")
                pass

    def _predict_models(self, X):
        """Get predictions from all models"""
        predictions = []
        for i, model in enumerate(self.models):
            try:
                if isinstance(model, GaussianProcessRegressor):
                    mu, sigma = model.predict(X, return_std=True)
                    predictions.append((mu, sigma))
                else:
                    pred = model.predict(X)
                    # For tree-based models, we use a simple estimate of uncertainty
                    # This is a simplification - in practice, more sophisticated methods would be used
                    pred = pred.reshape(-1, 1)
                    # Use the mean prediction as a proxy for mu and a small value for sigma
                    predictions.append((pred, np.full(pred.shape, 0.1)))
            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}")
                # Return dummy values
                predictions.append((np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))))
        return predictions

    def _acquisition_function_ei(self, X, model_predictions):
        """Expected Improvement acquisition function"""
        mu, sigma = model_predictions
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _acquisition_function_pi(self, X, model_predictions):
        """Probability of Improvement acquisition function"""
        mu, sigma = model_predictions
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Probability of Improvement calculation
        z = (y_min - mu) / sigma
        pi = norm.cdf(z)
        
        return pi.reshape(-1, 1)

    def _acquisition_function_thompson_sampling(self, X, model_predictions):
        """Thompson Sampling acquisition function"""
        mu, sigma = model_predictions
        # Sample from the posterior distribution
        samples = self.rng.normal(mu.flatten(), sigma.flatten())
        # Thompson sampling: minimize the sampled values (since we're minimizing)
        ts = -samples.reshape(-1, 1)
        return ts

    def _select_acquisition_function(self):
        """Select acquisition function based on recent performance"""
        if len(self.acq_func_history) < self.model_selection_window:
            # Use a simple round-robin approach initially
            return self.rng.integers(0, 3)
        
        # Look at recent performance of different acquisition functions
        recent_performance = self.acq_func_history[-self.model_selection_window:]
        if len(recent_performance) > 0:
            # Return the index of the best performing acquisition function
            return np.argmax(recent_performance)
        else:
            return 0  # Default to EI

    def _acquisition_function(self, X):
        """Select and compute acquisition function based on ensemble"""
        # Get predictions from all models
        model_predictions = self._predict_models(X)
        
        # Select which acquisition function to use
        acq_func_idx = self._select_acquisition_function()
        self.acq_func_type = acq_func_idx
        
        # Compute acquisition for each model's prediction
        acq_values = []
        for i, pred in enumerate(model_predictions):
            if acq_func_idx == 0:
                val = self._acquisition_function_ei(X, pred)
            elif acq_func_idx == 1:
                val = self._acquisition_function_pi(X, pred)
            else:  # Thompson Sampling
                val = self._acquisition_function_thompson_sampling(X, pred)
            acq_values.append(val)
        
        # Use weighted average of acquisition values from all models
        weights = np.array(self.model_weights)
        if len(weights) != len(acq_values):
            # If there's a mismatch, use equal weights
            weights = np.ones(len(acq_values)) / len(acq_values)
            
        weighted_acq = np.zeros_like(acq_values[0])
        for i in range(len(acq_values)):
            weighted_acq += weights[i] * acq_values[i]
        
        return weighted_acq

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Generate candidates using Sobol sequence
        n_starting_points = min(100 * self.dim, 5000)  # Increased from previous version
        
        # Generate starting points using Sobol sequence
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates with diversity consideration
        # Sort by acquisition value and select top ones
        sorted_indices = np.argsort(acq_values.flatten())[::-1]
        selected_indices = sorted_indices[:batch_size]
        selected_points = candidates[selected_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate all points in X
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y_vals
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        # Simple approach: if a model has been good recently, increase its weight
        # For simplicity, we'll just use a basic updating scheme here
        
        # In a more sophisticated version, we would evaluate the predictive performance 
        # of each model on held-out data or use cross-validation
        
        # Update weights to be proportional to recent performance
        total_performance = sum(self.model_performance)
        if total_performance > 0:
            self.model_weights = [p/total_performance for p in self.model_performance]
        else:
            # If no performance data, reset to equal weights
            self.model_weights = [1.0/len(self.models)] * len(self.models)
        
        # Reset performance tracking
        self.model_performance = [0.0] * len(self.models)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol sequence design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial models
        self._fit_models(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self._fit_models(self.X, self.y)
            
            # Update model weights periodically
            if self.n_evals % 10 == 0 and self.n_evals > self.n_init:
                self._update_model_weights()
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
