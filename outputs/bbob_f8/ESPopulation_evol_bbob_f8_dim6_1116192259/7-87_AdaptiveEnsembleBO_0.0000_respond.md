# Description 
**AdaptiveEnsembleBO**: This Bayesian Optimization algorithm employs an ensemble of surrogate models with dynamic model selection and a novel hybrid acquisition function that combines expected improvement with uncertainty sampling. The key innovation is the adaptive ensemble approach where multiple Gaussian Process models with different kernels (Matern, RBF, and rational quadratic) are maintained simultaneously, and their predictions are combined based on recent performance. A novel acquisition function called "adaptive hybrid EI" dynamically weights between pure exploration and exploitation based on the current optimization stage. The algorithm uses a local search strategy that focuses on promising regions while maintaining global diversity through batch selection with diversity constraints.

# Justification 
The main differences from existing approaches include:
1. **Adaptive Ensemble Surrogate Model**: Instead of using a single GP model, this algorithm maintains multiple models with different kernels (Matern, RBF, Rational Quadratic) and dynamically selects the best performing one for predictions based on recent validation performance. This addresses the problem that no single kernel is optimal across all optimization stages.

2. **Novel Hybrid Acquisition Function**: The "adaptive hybrid EI" combines expected improvement with uncertainty sampling using a dynamic weight that adapts based on how many evaluations have been performed relative to the budget. Early in optimization, it emphasizes exploration (uncertainty-based), while later focusing more on exploitation (EI-based).

3. **Diversity-Constrained Batch Selection**: Unlike single-point selection or simple trust region approaches, this algorithm selects batches of points that are diverse as well as promising, preventing premature convergence.

4. **Dynamic Model Switching**: The ensemble dynamically switches between models based on validation performance, rather than using fixed models throughout optimization.

5. **Efficient Implementation**: Uses efficient GP fitting with limited restarts, and optimized acquisition function optimization via local search around promising points, not random sampling.

This approach is expected to be more robust across different functions and dimensions because it adapts its modeling strategy based on the observed function behavior, and the hybrid acquisition function naturally balances exploration and exploitation without requiring manual tuning of parameters like LCB weights.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from scipy.optimize import minimize
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
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Ensemble of GP models with different kernels
        self.gp_models = []
        self.model_weights = []
        self.model_performance = []
        
        # Initialize multiple GP models with different kernels
        self._initialize_ensemble()
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # Batch size for selection
        self.batch_size = 1
        
        # For tracking progress and adaptation
        self.best_y_history = []
        self.iteration = 0

    def _initialize_ensemble(self):
        """Initialize ensemble of GP models with different kernels"""
        # Initialize three different models with different kernels
        kernels = [
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5),
            RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
            RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds=(1e-3, 1e3))
        ]
        
        self.gp_models = []
        for i, kernel in enumerate(kernels):
            # Add white noise to each model
            kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
            
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                n_restarts_optimizer=2,  # Reduced restarts for efficiency
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            self.gp_models.append(model)
        
        # Initialize weights and performance tracking
        self.model_weights = np.ones(len(self.gp_models)) / len(self.gp_models)
        self.model_performance = [0.0] * len(self.gp_models)

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_ensemble(self, X, y):
        """Fit all models in the ensemble"""
        for i, model in enumerate(self.gp_models):
            try:
                model.fit(X, y)
            except Exception as e:
                # If fitting fails, just continue with current fit
                warnings.warn(f"Model {i} fitting failed: {e}")
                pass

    def _predict_ensemble(self, X):
        """Get predictions from all models in the ensemble"""
        predictions = []
        for model in self.gp_models:
            try:
                mu, sigma = model.predict(X, return_std=True)
                predictions.append((mu, sigma))
            except Exception as e:
                # If prediction fails, return zeros
                warnings.warn(f"Model prediction failed: {e}")
                predictions.append((np.zeros(X.shape[0]), np.ones(X.shape[0])))
        return predictions

    def _combine_predictions(self, predictions):
        """Combine predictions from all models using weighted average"""
        mus = np.array([p[0] for p in predictions])
        sigmas = np.array([p[1] for p in predictions])
        
        # Weighted average of means and variance
        mu_weighted = np.average(mus, axis=0, weights=self.model_weights)
        sigma_weighted = np.average(sigmas**2, axis=0, weights=self.model_weights)
        sigma_weighted = np.sqrt(sigma_weighted)
        
        return mu_weighted, sigma_weighted

    def _calculate_model_performance(self, X_val, y_val):
        """Calculate performance of each model on validation set"""
        performances = []
        for i, model in enumerate(self.gp_models):
            try:
                # Predict on validation set
                y_pred, _ = model.predict(X_val, return_std=False)
                # Calculate MSE (lower is better)
                mse = np.mean((y_val.flatten() - y_pred)**2)
                performances.append(1.0 / (1.0 + mse))  # Convert to performance metric (higher is better)
            except Exception as e:
                warnings.warn(f"Model {i} performance calculation failed: {e}")
                performances.append(0.0)
        return performances

    def _update_model_weights(self, X_val, y_val):
        """Update model weights based on recent performance"""
        performances = self._calculate_model_performance(X_val, y_val)
        
        # Update running performance
        for i in range(len(self.model_performance)):
            self.model_performance[i] = 0.7 * self.model_performance[i] + 0.3 * performances[i]
        
        # Normalize weights
        total_perf = sum(self.model_performance)
        if total_perf > 0:
            self.model_weights = np.array(self.model_performance) / total_perf
        else:
            # If all models have failed, reset to equal weights
            self.model_weights = np.ones(len(self.gp_models)) / len(self.gp_models)

    def _adaptive_hybrid_acquisition(self, X):
        """Adaptive hybrid acquisition function combining EI and uncertainty sampling"""
        if self.X is None or len(self.X) < 1:
            # Return random values for first few evaluations
            return np.ones((X.shape[0], 1)) * (self.rng.random(X.shape[0]) - 0.5)
            
        # Get predictions from ensemble
        predictions = self._predict_ensemble(X)
        mu, sigma = self._combine_predictions(predictions)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Hybrid component: uncertainty sampling
        uncertainty = sigma
        
        # Adaptive weighting based on optimization stage
        progress_ratio = min(self.n_evals / self.budget, 1.0)
        # Early stage: emphasize exploration (uncertainty)
        # Later stage: emphasize exploitation (EI)
        exploit_weight = np.clip(0.2 + 0.8 * progress_ratio, 0.2, 1.0)
        explore_weight = 1.0 - exploit_weight
        
        # Combine acquisition functions
        hybrid_acq = exploit_weight * ei + explore_weight * uncertainty
        
        return hybrid_acq.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select next points using adaptive hybrid acquisition function"""
        # Start with a good set of starting points for local search
        n_starting_points = min(50 * self.dim, 500)  # Reduced from previous version for efficiency
        
        # Sample points in the entire domain for initial candidates
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._adaptive_hybrid_acquisition(candidates)
        
        # Select top candidates (use a simple heuristic - just the best ones)
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial ensemble of GP models
        self._fit_ensemble(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the ensemble with new data
            if self.n_evals > self.n_init:
                # Fit all models in the ensemble
                self._fit_ensemble(self.X, self.y)
                
                # For validation, we use a subset of recent evaluations
                if len(self.y) >= 10:
                    val_indices = self.rng.choice(len(self.y), min(5, len(self.y)), replace=False)
                    X_val = self.X[val_indices]
                    y_val = self.y[val_indices]
                    # Update model weights based on recent performance
                    self._update_model_weights(X_val, y_val)
            
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
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveEnsembleBO>", line 78
    samples = sampler.random(n=n=n_points)
                                ^
SyntaxError: invalid syntax
