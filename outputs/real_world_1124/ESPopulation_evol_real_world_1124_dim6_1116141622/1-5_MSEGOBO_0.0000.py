from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

class MSEGOBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, dim * 2)  # Initial points based on dimensionality

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Acquisition function parameters
        self.kappa = 2.576  # For UCB acquisition, corresponds to 99% confidence
        self.xi = 0.01      # For EI acquisition parameter
        
        # Local search parameters
        self.local_search_freq = 0.2  # Frequency of local search (as percentage of budget)
        self.local_search_max_iter = 30
        self.acquisition_candidates = 100  # Number of candidates for acquisition function

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with automatic relevance determination
        # Use a composite kernel: RBF + Constant (to capture noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # For better numerical stability, we add a small noise term
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=1,  # Few restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        # Fit the model
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, best_y):
        # Use a simplified acquisition function to reduce computational overhead
        # Only compute UCB for efficiency
        
        # Get predictions from the GP
        mu, sigma = model.predict(X, return_std=True)
        
        # Compute Upper Confidence Bound (UCB) 
        ucb = mu + self.kappa * sigma
        
        return ucb.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a more budget-conscious approach
        # Generate candidates for selection with careful budget management
        
        # Use adaptive candidate size based on remaining budget and current evaluation count
        remaining_budget = self.budget - self.n_evals
        max_candidates = min(100, max(20, int(remaining_budget * 0.5)))
        
        if max_candidates < batch_size:
            batch_size = max_candidates
            
        # Sample points for acquisition function evaluation
        candidates = self._sample_points(max_candidates)
        
        # If we have no data yet, just return sampled points
        if self.X is None or len(self.X) == 0:
            return candidates[:batch_size]
            
        # Fit model and evaluate acquisition function
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidates, model, np.min(self.y))
        
        # Select top points based on acquisition values
        top_indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y_values = []
        for x in X:
            y_values.append(func(x))
        
        y = np.array(y_values).reshape(-1, 1)
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

    def _gradient_based_local_search(self, func, x_start, max_iter=30):
        """Perform gradient-based local search to refine a point"""
        # Simple gradient descent approach
        learning_rate = 0.01
        x = x_start.copy()
        
        for _ in range(max_iter):
            # Evaluate current function value
            f_current = func(x)
            
            # Estimate gradient using finite differences
            epsilon = 1e-4
            grad = np.zeros_like(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                f_plus = func(x_plus)
                grad[i] = (f_plus - f_current) / epsilon
            
            # Update step (with bounds checking)
            x_new = x - learning_rate * grad
            
            # Project back to bounds
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            
            # If improvement, accept the new point
            if func(x_new) < f_current:
                x = x_new
            else:
                break  # No improvement, stop
                
        return x

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Track best solution
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Determine batch size - larger early on for more exploration
            remaining_budget = self.budget - self.n_evals
            
            # Calculate appropriate batch size based on remaining budget and progress
            if remaining_budget <= 20:
                batch_size = 1
            elif remaining_budget <= 50:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, max(1, int((remaining_budget / self.budget) * 10)))
            
            # Ensure we don't exceed budget
            if self.n_evals + batch_size > self.budget:
                batch_size = self.budget - self.n_evals
                
            if batch_size <= 0:
                break
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_values)
            
            # Update best solution found so far
            current_best_y = np.min(self.y)
            if current_best_y < best_y:
                best_y = current_best_y
                best_x = self.X[np.argmin(self.y)]
                
            # Perform local search on some points periodically (adaptive)
            if self.n_evals > 0 and (self.n_evals % max(1, int(self.budget * self.local_search_freq))) == 0:
                # Select a few good candidates for local search
                model = self._fit_model(self.X, self.y)
                candidate_indices = np.argsort(self.y.flatten())[:min(2, len(self.X))]
                
                for idx in candidate_indices:
                    x_candidate = self.X[idx]
                    refined_x = self._gradient_based_local_search(func, x_candidate, self.local_search_max_iter)
                    refined_y = func(refined_x)
                    
                    # If improvement, update the dataset
                    if refined_y < best_y:
                        best_y = refined_y
                        best_x = refined_x
                        # Add to our data points
                        self._update_eval_points(refined_x.reshape(1, -1), np.array([[refined_y]]))
        
        return best_y, best_x
