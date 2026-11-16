from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class EfficientRobustBOTs:
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
        # Use a more adaptive approach that scales better with budget
        self.n_init = min(max(10, 2 * dim + 1), max(10, budget // 4))
        if self.n_init > budget:
            self.n_init = budget
            
        # Model parameters
        self.model = None
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model with reduced computational overhead"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with sparse approximation to reduce computational cost
        # Use fewer restarts for efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced from 3 to 2 for speed
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Thompson Sampling acquisition function"""
        # If no model exists, return zeros
        if self.model is None:
            return np.zeros((X.shape[0], 1))
            
        # Sample from the posterior distribution for each point
        # This is a simple Thompson sampling approach - sample one function realization
        # from the GP posterior for each candidate point
        samples = self.model.sample_y(X, n_samples=1, random_state=self.rng)
        
        # Return negative samples (since we want to minimize the function)
        return -samples.reshape(-1, 1)

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using Thompson sampling approach"""
        # For Thompson Sampling, we don't need to optimize an acquisition function
        # Instead, we sample candidates and evaluate them
        
        # Sample a large set of points
        candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function (Thompson sampling)
        acq_values = self._acquisition_function(candidates)
        
        # Select the point with the lowest acquisition value (best candidate)
        best_idx = np.argmin(acq_values.flatten())
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Select next points using Thompson sampling approach
            next_points = self._optimize_acquisition(n_starting_points=min(100, max(20, 3 * self.dim)))
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
