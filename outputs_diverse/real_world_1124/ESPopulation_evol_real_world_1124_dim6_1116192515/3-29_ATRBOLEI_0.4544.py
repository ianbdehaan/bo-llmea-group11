from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
import warnings

class ATRBOLEI:
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
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.trust_region_center = None
        self.success_count = 0
        self.failure_count = 0
        self.min_trust_region = 0.05
        self.max_trust_region = 2.0

    def _sample_points(self, n_points):
        # Use Sobol sequence sampling for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, seed=self.rng, scramble=True)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a sparse Gaussian Process with ARD Matern kernel
        # Use a small subset of data for training to keep computational cost low
        n_train = min(50, max(20, int(0.5 * len(X))))
        
        # If we have enough points, use a subset for fitting (sparse GP)
        if len(X) > 30:
            # Select a random subset of points for training to keep it fast
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
        # Use log expected improvement (LEI) as the acquisition function
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
        # Log Expected Improvement (LEI)
        best_y = np.min(self.y)
        
        # Handle case where sigma is zero (no uncertainty)
        mask = sigma > 0
        lei_values = np.zeros_like(mu)
        
        if np.any(mask):
            # For numerical stability, we add a small epsilon to avoid log(0)
            epsilon = 1e-8
            
            # Calculate improvement over best_y
            improvement = best_y - mu[mask]
            
            # Handle case where improvement is positive (we want to improve)
            # We compute the log expected improvement using the standard formula
            z = improvement / sigma[mask]
            
            # Use the mathematical definition of LEI
            # LEI = log(E[I(y < y_best)]) where I is indicator function
            # This can be computed as: log(Phi(-z) + (improvement/sigma) * phi(-z))
            # But we use a more numerically stable approach
            
            # Standard EI computation for reference:
            # ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            # LEI is the logarithm of this, but we need to be careful with numerical issues
            # We compute it directly using the relationship:
            # LEI = log(E[I(y < y_best)]) where I(y < y_best) = Phi((y_best - mu)/sigma)
            
            # For practical implementation, we use the standard EI and then take log
            # But ensure we don't get negative values in the logarithm
            
            ei_values = improvement * norm.cdf(z) + sigma[mask] * norm.pdf(z)
            
            # Take log of positive values, set negative to a small negative value for numerical stability
            lei_values[mask] = np.log(np.maximum(ei_values, epsilon))
        
        return lei_values.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select points using a hybrid approach:
        # 1. First try to find the best point with acquisition function
        # 2. Then use local search around promising regions
        
        # Sample candidates for optimization
        n_candidates = min(1000, max(50, self.budget - self.n_evals))
        
        # Get points from current trust region if we have a center
        if self.trust_region_center is not None:
            # Generate points around the trust region center
            candidates = self._sample_points(n_candidates)
            
            # Apply trust region constraint: only consider points within radius
            distances = np.linalg.norm(candidates - self.trust_region_center, axis=1)
            valid_indices = np.where(distances <= self.trust_region_radius)[0]
            if len(valid_indices) > 0:
                candidates = candidates[valid_indices]
            
            # If not enough candidates in trust region, expand the search
            if len(candidates) < batch_size:
                additional_candidates = self._sample_points(batch_size - len(candidates))
                candidates = np.vstack([candidates, additional_candidates])
        else:
            # First few iterations: random sampling with trust region expansion
            candidates = self._sample_points(n_candidates)
        
        # Limit candidates to reasonable size
        if len(candidates) > n_candidates:
            candidates = candidates[:n_candidates]
            
        # Evaluate acquisition function on candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select best points using top-k selection
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points

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
            
        # Update trust region based on success of recent evaluations
        if len(self.y) > 10:
            # Check if we've made progress recently (last 5 evaluations)
            recent_improvement = np.min(self.y[-5:]) < np.min(self.y[:-5]) - 0.01
            if recent_improvement:
                self.success_count += 1
                self.failure_count = 0
                # Shrink trust region if we're making progress
                if self.trust_region_radius > self.min_trust_region:
                    self.trust_region_radius *= 0.95
            else:
                self.failure_count += 1
                self.success_count = 0
                # Expand trust region if we're not making progress
                if self.trust_region_radius < self.max_trust_region:
                    self.trust_region_radius *= 1.1
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Set trust region center to the best point found so far
        best_idx = np.argmin(self.y)
        self.trust_region_center = self.X[best_idx]
        
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
            
            # Update trust region center based on new best point
            if len(self.y) > 0:
                best_idx = np.argmin(self.y)
                self.trust_region_center = self.X[best_idx]
                
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
