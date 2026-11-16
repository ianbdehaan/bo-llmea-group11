from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution
import warnings

class DiversityEnhancedBO:
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
        self.n_init = min(max(15, 3 * dim + 1), budget // 3)
        if self.n_init > budget:
            self.n_init = budget
            
        # GP model parameters
        self.gp_model = None
        
        # Batch selection parameters
        self.batch_size = min(5, max(1, dim // 2))
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # Performance tracking for switch from global to local search
        self.best_y_history = []
        self.iteration = 0
        self.switch_threshold = max(5, budget // 10)  # Switch after this many iterations
        
        # Diversity enhancement parameters
        self.diversity_weight = 0.3
        self.diversity_radius_factor = 0.1  # Relative to search space size
        
        # For global-to-local switch
        self.global_exploitation_ratio = 0.7
        self.exploitation_schedule = None

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model with adaptive noise"""
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
        """Expected Improvement acquisition function with adaptive exploration"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei.reshape(-1, 1)

    def _diversity_adjusted_acquisition(self, X):
        """Compute acquisition function with diversity penalty"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get basic acquisition values
        ei = self._acquisition_function(X).flatten()
        
        # If we have previous points, compute diversity penalties
        if self.X is not None and len(self.X) > 0:
            # Compute pairwise distances between candidates and existing points
            distances = cdist(X, self.X)
            
            # Compute diversity score (minimum distance to any existing point)
            min_distances = np.min(distances, axis=1)
            
            # Normalize distances by search space size
            search_space_size = np.max(self.bounds[1]) - np.min(self.bounds[0])
            normalized_distances = min_distances / search_space_size
            
            # Apply diversity penalty (higher penalty for closer points)
            diversity_penalty = 1.0 / (1.0 + self.diversity_weight * normalized_distances)
            
            # Adjust acquisition values with diversity
            adjusted_ei = ei * diversity_penalty
        else:
            adjusted_ei = ei
            
        return adjusted_ei.reshape(-1, 1)

    def _select_next_points(self):
        """Select the next points to evaluate using batch selection with diversity"""
        # Determine how many candidates to generate for optimization
        n_candidates = min(50 * self.dim, 1000)
        
        # Generate candidates across the search space
        candidates = self._sample_points(n_candidates)
        
        # Compute diversity-adjusted acquisition function values
        acq_values = self._diversity_adjusted_acquisition(candidates).flatten()
        
        # Use simulated annealing approach to select diverse batch
        # Start with greedy selection and then use simulated annealing to improve diversity
        
        # Get top candidates by acquisition value
        top_indices = np.argsort(acq_values)[-self.batch_size:]
        
        # Refine the selection using a simple simulated annealing-like approach
        selected_indices = list(top_indices)
        
        # For small batches, try to increase diversity by swapping elements
        if self.batch_size > 1:
            for _ in range(5):  # Few iterations of swapping
                if len(selected_indices) < 2:
                    break
                    
                # Try swapping two random points
                i1, i2 = self.rng.choice(len(selected_indices), size=2, replace=False)
                old_score = acq_values[selected_indices[i1]] + acq_values[selected_indices[i2]]
                
                # Temporarily swap and see if we can improve diversity while maintaining acquisition value
                temp_indices = selected_indices.copy()
                temp_indices[i1], temp_indices[i2] = temp_indices[i2], temp_indices[i1]
                
                # Compute new scores with diversity adjustment
                new_scores = self._diversity_adjusted_acquisition(candidates[temp_indices])
                new_score = np.sum(new_scores)
                
                if new_score > old_score:
                    selected_indices = temp_indices
        
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

    def _switch_to_local_search(self):
        """Switch from global to local search based on progress"""
        # Switch if we've made significant progress or reached threshold
        if len(self.best_y_history) < 2:
            return False
            
        # Check if we've made substantial improvement recently
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        # Switch to local search after a certain number of iterations or significant progress
        return self.iteration >= self.switch_threshold or improvement > 0.05

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
            
            # Check for switch to local search
            if self._switch_to_local_search():
                # When switching, we reduce exploration and focus on exploitation
                pass  # The diversity adjustment naturally decreases as we converge
            
            # Select next points using batch selection with diversity
            next_points = self._select_next_points()
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            self.iteration += 1
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
