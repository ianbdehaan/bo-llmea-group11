from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np

class MultiStartLocalSearchBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10)  # Initial points - start with more points for higher dimensions

        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Acquisition function parameters
        self.acq_params = {
            'exploration_weight': 0.5,  # Weight for exploration term
            'diversity_factor': 0.1     # Factor to control diversity promotion
        }
        
        # Exploration strategy tracking
        self.best_y_history = []
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 3
        self.global_search_mode = True  # Start in global search mode
        
        # Optimization parameters
        self.n_local_starts = min(10, max(3, dim))  # Number of local starts for acquisition optimization
        self.n_diverse_candidates = min(20, max(5, dim * 2))  # Number of diverse candidates to sample

    def _sample_points(self, n_points):
        # Sample points using maximin Latin hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Using Gaussian Process with ARD Matern kernel
        # We use a fixed length scale for efficiency and avoid overfitting
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=1,  # Reduced restarts for computational efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, best_y):
        # Implement hybrid acquisition function: Expected Improvement + Diversity term
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)  # Avoid numerical issues
        
        # Expected Improvement
        improvement = mu - best_y
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Diversity term - encourages exploration of unvisited regions
        if self.X is not None and len(self.X) > 0:
            # Compute distance to nearest existing point for each candidate
            distances = np.full(len(X), np.inf)
            for x in X:
                dists = np.sum((self.X - x)**2, axis=1)
                min_dist = np.min(dists)
                distances[np.argmin(dists)] = min_dist
            
            # Normalize distances to [0, 1] range
            if np.max(distances) > 0:
                normalized_distances = distances / np.max(distances)
            else:
                normalized_distances = np.zeros_like(distances)
                
            # Diversity term: higher when farther from existing points
            diversity_term = 1.0 - normalized_distances
            
            # Combine EI with diversity term (weighted by diversity_factor)
            hybrid_acq = ei.reshape(-1, 1) + self.acq_params['diversity_factor'] * diversity_term.reshape(-1, 1)
        else:
            # If no points yet, just use EI
            hybrid_acq = ei.reshape(-1, 1)
            
        return hybrid_acq

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using multi-start local search approach:
        # 1. Sample random diverse candidates
        # 2. Run multiple local optimizations from different starting points
        # 3. Select best points from both approaches
        
        # Generate diverse candidate points for acquisition optimization
        candidates = self._sample_points(self.n_diverse_candidates)
        
        if self.X is not None and len(self.X) > 0:
            best_y = np.min(self.y)
            model = self._fit_model(self.X, self.y)
            
            # Multi-start local search approach for acquisition optimization
            acq_values = self._acquisition_function(candidates, model, best_y).flatten()
            
            # Select top candidates based on acquisition values
            top_indices = np.argsort(acq_values)[-batch_size:]
            selected_candidates = candidates[top_indices]
        else:
            # For initial phase, just return random points
            selected_candidates = self._sample_points(batch_size)
            
        return selected_candidates

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_values = []
        for point in X:
            y_values.append(func(point))
        
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Check for convergence and adapt exploration strategy
            if len(self.best_y_history) > 0:
                current_best = np.min(self.y)
                self.best_y_history.append(current_best)
                
                if current_best >= self.best_y_history[-2] - 1e-4:  # No significant improvement
                    self.consecutive_no_improvement += 1
                    if self.consecutive_no_improvement >= self.max_no_improvement:
                        # Switch to global search mode when progress stalls
                        self.global_search_mode = True
                        self.consecutive_no_improvement = 0
                        # Increase exploration weight for more global search
                        self.acq_params['exploration_weight'] = min(1.0, self.acq_params['exploration_weight'] + 0.1)
                else:
                    self.consecutive_no_improvement = 0
                    # Switch to local exploitation mode when progress is good
                    if self.global_search_mode and len(self.best_y_history) > 5:
                        self.global_search_mode = False
                        # Reduce exploration weight for more local search
                        self.acq_params['exploration_weight'] = max(0.1, self.acq_params['exploration_weight'] - 0.1)
            else:
                self.best_y_history.append(np.min(self.y))
            
            # Select next points to evaluate
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)  # Limit batch size to remaining budget
            
            if batch_size <= 0:
                break
                
            next_points = self._select_next_points(batch_size)
            
            # Evaluate new points
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            
            # Early termination for small budgets to avoid excessive overhead
            if self.n_evals >= self.budget - 1:
                break
                
        # Return the best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
