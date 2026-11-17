from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

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
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), max(10, budget // 4))
        if self.n_init > budget:
            self.n_init = budget
            
        # Model parameters
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.best_model = None
        
        # Acquisition function parameters
        self.acq_weights = [0.4, 0.3, 0.3]  # EI, PI, Thompson Sampling
        self.noise_level = 1e-6
        
        # Batch parameters
        self.batch_size = 1
        self.batch_growth_factor = 1.2
        
        # For local quadratic model
        self.local_model_points = None
        self.local_model_weights = None
        
        # For tracking progress
        self.history = []
        
    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit Gaussian Process model with ARD Matern kernel"""
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _fit_rf_model(self, X, y):
        """Fit Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        
        model.fit(X, y.ravel())
        return model

    def _fit_local_quadratic_model(self, X, y):
        """Fit a local quadratic model around the best point"""
        if len(y) < 3:
            return None
            
        # Find the best point
        best_idx = np.argmin(y)
        best_x = X[best_idx]
        
        # Get nearest neighbors for local model
        distances = cdist([best_x], X)[0]
        neighbor_indices = np.argsort(distances)[:min(10, len(X))]
        
        # Fit a simple quadratic model using these points
        X_local = X[neighbor_indices]
        y_local = y[neighbor_indices]
        
        try:
            # Simple linear regression for local model (could be extended to quadratic)
            # Using least squares approach
            A = np.ones((len(X_local), 1))
            for i in range(self.dim):
                A = np.column_stack([A, X_local[:, i]])
                
            coeffs = np.linalg.lstsq(A, y_local, rcond=None)[0]
            
            def local_predict(x):
                x_vec = np.array([1] + list(x))
                return np.dot(x_vec, coeffs)
                
            return local_predict
        except:
            return None

    def _update_models(self):
        """Update all models with current data"""
        # Update GP model
        self.models['gp'] = self._fit_gp_model(self.X, self.y)
        
        # Update RF model  
        self.models['rf'] = self._fit_rf_model(self.X, self.y)
        
        # Update local quadratic model
        self.models['quad'] = self._fit_local_quadratic_model(self.X, self.y)

    def _get_predictions(self, X):
        """Get predictions from all models"""
        predictions = {}
        
        # GP prediction
        if 'gp' in self.models and self.models['gp'] is not None:
            try:
                mu_gp, sigma_gp = self.models['gp'].predict(X, return_std=True)
                predictions['gp'] = (mu_gp.reshape(-1, 1), sigma_gp.reshape(-1, 1))
            except:
                predictions['gp'] = None
        else:
            predictions['gp'] = None
            
        # RF prediction
        if 'rf' in self.models and self.models['rf'] is not None:
            try:
                pred_rf = self.models['rf'].predict(X)
                # For RF, we'll use a simple estimate of uncertainty
                predictions['rf'] = (pred_rf.reshape(-1, 1), np.ones((len(pred_rf), 1)) * 0.1)
            except:
                predictions['rf'] = None
        else:
            predictions['rf'] = None
            
        # Quadratic model prediction
        if 'quad' in self.models and self.models['quad'] is not None:
            try:
                pred_quad = np.array([self.models['quad'](x) for x in X])
                predictions['quad'] = (pred_quad.reshape(-1, 1), np.ones((len(pred_quad), 1)) * 0.1)
            except:
                predictions['quad'] = None
        else:
            predictions['quad'] = None
            
        return predictions

    def _ensemble_prediction(self, X):
        """Get ensemble prediction using weighted average of models"""
        predictions = self._get_predictions(X)
        
        # Get all valid predictions
        valid_preds = []
        weights = []
        
        for model_name, pred in predictions.items():
            if pred is not None:
                mu, sigma = pred
                valid_preds.append(mu)
                weights.append(self.model_weights.get(model_name, 1.0))
                
        if not valid_preds:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average of predictions
        ensemble_mu = np.average(np.array(valid_preds), axis=0, weights=weights)
        # Simple combination of variances (assuming independence)
        ensemble_sigma = np.sqrt(np.average(np.array([sigma**2 for sigma in valid_preds]), 
                                           axis=0, weights=weights))
        
        return ensemble_mu, ensemble_sigma

    def _acquisition_function(self, X):
        """Hybrid acquisition function combining EI, PI and Thompson Sampling"""
        # Get ensemble predictions
        mu, sigma = self._ensemble_prediction(X)
        
        # Find best observed value
        y_min = np.min(self.y)
        
        # Calculate components
        ei_vals = np.zeros((X.shape[0], 1))
        pi_vals = np.zeros((X.shape[0], 1))
        thompson_vals = np.zeros((X.shape[0], 1))
        
        # Expected Improvement (EI) - use standard formula
        if sigma is not None and len(sigma) > 0:
            sigma = np.maximum(sigma, 1e-8)
            improvement = y_min - mu - 1e-6  # Small jitter to avoid numerical issues
            z = improvement / sigma
            ei_vals = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
        # Probability of Improvement (PI)
        if sigma is not None and len(sigma) > 0:
            sigma = np.maximum(sigma, 1e-8)
            z = (y_min - mu) / sigma
            pi_vals = norm.cdf(z)
            
        # Thompson Sampling - sample from GP posterior
        if 'gp' in self.models and self.models['gp'] is not None:
            try:
                samples = self.models['gp'].sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
                thompson_vals = (y_min - samples).reshape(-1, 1)
            except:
                # Fallback to standard approach if sampling fails
                thompson_vals = np.zeros((X.shape[0], 1))
        
        # Combine using adaptive weights
        acq_vals = (
            self.acq_weights[0] * ei_vals +
            self.acq_weights[1] * pi_vals + 
            self.acq_weights[2] * thompson_vals
        )
        
        return acq_vals

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using a hybrid approach"""
        # Sample initial candidates with better distribution
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points to start optimization from
        top_indices = np.argsort(acq_values.flatten())[-max(5, min(self.dim, 10)):]
        starting_points = candidates[top_indices]
        
        # Use local optimization around the best candidates
        best_acq_val = -np.inf
        best_point = None
        
        for x0 in starting_points:
            try:
                # Simple bounded L-BFGS optimization with restarts
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    options={'maxiter': 30}
                )
                
                if result.success:
                    acq_val = -result.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = result.x
                        
            except:
                # If optimization fails, continue with current best
                pass
        
        # Fallback to the best candidate from sampling if optimization fails
        if best_point is None:
            best_idx = np.argmax(acq_values.flatten())
            best_point = candidates[best_idx]
            
        return best_point.reshape(1, -1)

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

    def _adapt_model_weights(self):
        """Adapt model weights based on performance"""
        if len(self.history) < 3:
            # Initialize weights for first few evaluations
            self.model_weights = {'gp': 0.4, 'rf': 0.3, 'quad': 0.3}
            return
            
        # Simple performance tracking based on recent improvement
        recent_performance = self.history[-5:] if len(self.history) >= 5 else self.history
        
        if len(recent_performance) > 1:
            # Calculate improvement in best value over last few steps
            improvements = []
            for i in range(len(recent_performance)-1):
                prev_best = min(recent_performance[i])
                curr_best = min(recent_performance[i+1])
                improvements.append(prev_best - curr_best)
            
            if len(improvements) > 0:
                avg_improvement = np.mean(improvements)
                
                # Adjust weights based on recent performance
                # If GP is performing well, increase its weight
                if self.model_performance.get('gp', 0) > 0.5:
                    self.model_weights['gp'] = min(0.8, self.model_weights['gp'] + 0.05)
                else:
                    self.model_weights['gp'] = max(0.1, self.model_weights['gp'] - 0.05)
                    
                # RF and quad weights also adjusted accordingly
                self.model_weights['rf'] = 1.0 - self.model_weights['gp'] - self.model_weights['quad']
                self.model_weights['quad'] = 1.0 - self.model_weights['gp'] - self.model_weights['rf']

    def _calculate_batch_size(self):
        """Calculate dynamic batch size"""
        # Start with 1, grow logarithmically
        batch_size = max(1, int(np.log(self.n_evals + 1) * self.batch_growth_factor))
        return min(batch_size, 5)  # Cap at reasonable size

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Initialize model weights
        self.model_weights = {'gp': 0.4, 'rf': 0.3, 'quad': 0.3}
        
        # Fit initial models
        self._update_models()
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self._update_models()
            
            # Update dynamic batch size
            self.batch_size = self._calculate_batch_size()
            
            # Update model weights based on performance
            self._adapt_model_weights()
            
            # Select next points using acquisition function optimization
            # For batch selection, we'll optimize one point at a time for simplicity
            next_points = self._optimize_acquisition(n_starting_points=min(50, max(10, 2 * self.dim)))
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Store history for adaptive weighting
            self.history.append(self.y.flatten().copy())
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
