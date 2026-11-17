from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings

class StochasticMultiStartBO:
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
            
        # GP model parameters
        self.gp_model = None
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        
        # Exploration control
        self.uncertainty_threshold = 0.5  # Starting uncertainty threshold
        self.uncertainty_decay_rate = 0.95  # How fast uncertainty threshold decays
        
        # Multi-start parameters
        self.n_starting_points = min(100 * self.dim, 1000)  # Number of starting points for acquisition optimization
        self.min_starting_points = 10  # Minimum number of starting points
        self.max_starting_points = 500  # Maximum number of starting points
        
        # Performance tracking
        self.best_y_history = []
        self.last_improvement_iter = 0

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Entropy-based acquisition function that balances exploration and exploitation"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Compute entropy-based acquisition (information gain)
        # We use the variance as a proxy for information gain
        # Higher variance = more uncertainty = more exploration value
        entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)  # Differential entropy
        
        # Add uncertainty threshold to control exploration
        # Points with high uncertainty (variance) get higher acquisition values
        acq_values = sigma - self.uncertainty_threshold
        
        # Ensure non-negative acquisition values
        acq_values = np.maximum(acq_values, 0)
        
        return acq_values.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using stochastic multi-start optimization"""
        # Generate candidates using Latin hypercube sampling
        candidates = self._sample_points(self.n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best candidates based on acquisition values
        # Use a stochastic selection to avoid getting stuck in local optima
        sorted_indices = np.argsort(acq_values.flatten())[::-1]
        selected_indices = sorted_indices[:batch_size]
        
        # For better diversity, we also consider some random points among the top candidates
        if batch_size < len(selected_indices):
            # Add some randomness to selection to avoid deterministic behavior
            random_indices = self.rng.choice(len(selected_indices), 
                                           size=min(2, len(selected_indices)), 
                                           replace=False)
            selected_indices = np.concatenate([selected_indices[:batch_size-2], 
                                             selected_indices[random_indices]])
        
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
            
            # Update uncertainty threshold to adapt exploration
            if len(self.best_y_history) > 1:
                improvement = (self.best_y_history[-2] - self.best_y_history[-1]) / max(1e-8, abs(self.best_y_history[-2]))
                if improvement > 0.01:  # If significant improvement, reduce uncertainty threshold
                    self.uncertainty_threshold *= self.uncertainty_decay_rate
            
            # Early exit when budget is nearly exhausted
            if self.n_evals + 2 > self.budget:
                break
                
            # Select next points using stochastic multi-start acquisition optimization
            next_points = self._select_next_points(1)
            
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
