from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
import warnings

class EMBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling
        
        # Exploration schedule
        self.exploration_weight_schedule = None
        
        # Surrogate model parameters
        self.model_fit_threshold = 30

    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for better space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a sparse Gaussian Process with ARD Matern kernel
        # Use a small subset of data for training to keep computational cost low
        
        # If we have enough points, use a subset for fitting (sparse GP)
        if len(X) > self.model_fit_threshold:
            # Select a random subset of points for training to keep it fast
            n_train = min(50, max(20, int(0.5 * len(X))))
            train_indices = self.rng.choice(len(X), size=n_train, replace=False)
            X_train = X[train_indices]
            y_train = y[train_indices]
        else:
            X_train = X
            y_train = y
            
        # Use ARD Matern kernel with automatic relevance detection
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # Fit GP with noise estimation and optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_train, y_train)
        return gp

    def _acquisition_function(self, X):
        # Use an ensemble of acquisition functions weighted by their performance
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
        # Expected Improvement (EI)
        best_y = np.min(self.y)
        ei_values = np.zeros_like(mu)
        
        # Handle case where sigma is zero (no uncertainty)
        mask = sigma > 0
        if np.any(mask):
            improvement = best_y - mu[mask]
            z = improvement / sigma[mask]
            ei_values[mask] = improvement * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        # Lower Confidence Bound (LCB)
        # Adaptive exploration weight that decreases over time
        if self.exploration_weight_schedule is None:
            # Initialize exploration schedule
            self.exploration_weight_schedule = np.linspace(2.0, 0.5, max(10, int(self.budget * 0.3)))
        
        # Determine current exploration weight based on progress
        progress_ratio = min(1.0, self.n_evals / self.budget)
        if progress_ratio < 0.3:
            # Early phase: more exploration
            kappa = 2.0
        elif progress_ratio < 0.7:
            # Middle phase: moderate exploration
            kappa = 1.0
        else:
            # Late phase: more exploitation
            kappa = 0.5
            
        lcb = mu - kappa * sigma
        
        # Thompson sampling: sample from posterior distribution
        samples = gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        thompson = -samples.flatten()  # Negative because we minimize
        
        # Combine acquisition functions with weights
        acq_values = (
            self.acq_func_weights[0] * ei_values.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _optimize_acquisition_function(self, X_init, n_restarts=10):
        """Optimize acquisition function using multiple random restarts"""
        if len(X_init) == 0:
            return np.array([])
            
        best_acq_val = -np.inf
        best_point = None
        
        # Use multiple random restarts to find better candidates
        for _ in range(n_restarts):
            # Random initialization within bounds
            init_point = self._sample_points(1)[0]
            
            try:
                # Simple local optimization using scipy's differential evolution
                res = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    self.bounds.T,
                    seed=int(self.rng.integers(2**31 - 1)),
                    maxiter=50,
                    popsize=5
                )
                
                if res.success:
                    acq_val = -res.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_point = res.x
            except Exception:
                # If optimization fails, keep current point
                continue
                
        return best_point.reshape(1, -1) if best_point is not None else X_init[:1]

    def _select_next_points(self, batch_size):
        # Select points using multi-start approach with better exploration
        
        # Sample candidates for optimization
        n_candidates = min(500, max(20, self.budget - self.n_evals))
        
        # Get points from current trust region if we have a center
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function on candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points using top-k selection with multi-start optimization
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        
        # For each selected point, perform local optimization to refine it
        selected_points = []
        for idx in top_indices:
            init_point = candidates[idx:idx+1]
            refined_point = self._optimize_acquisition_function(init_point)
            selected_points.append(refined_point[0])
            
        return np.array(selected_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if len(X) == 0:
            return np.array([])
            
        y = []
        for point in X:
            try:
                val = func(point)
                y.append(val)
            except Exception as e:
                # Handle function evaluation errors by returning large value
                y.append(1e10)
                
        y = np.array(y).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, evaluate more points in parallel
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate these points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
