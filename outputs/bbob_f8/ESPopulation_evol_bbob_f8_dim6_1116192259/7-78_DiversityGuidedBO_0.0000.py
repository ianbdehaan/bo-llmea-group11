from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class DiversityGuidedBO:
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
        
        # Multi-armed bandit for strategy selection
        self.strategy_weights = np.array([0.33, 0.33, 0.34])  # random, local, global
        self.strategy_rewards = np.zeros(3)
        self.strategy_counts = np.zeros(3)
        self.strategy_names = ['random', 'local', 'global']
        
        # Diversity parameters
        self.diversity_threshold = 0.1
        self.diversity_penalty_factor = 0.5
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # Population-based optimization
        self.population_size = min(20, max(5, dim))
        self.population = None
        self.population_scores = None
        
        # For batch selection
        self.batch_size = 1
        
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
            n_restarts_optimizer=1,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _diversity_adjusted_ei(self, X):
        """Expected Improvement with diversity penalty"""
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
        
        # Apply diversity penalty if points are too close to existing ones
        if self.X is not None and len(self.X) > 0:
            # Compute distances between new points and existing points
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            
            # Apply penalty for points that are too close
            penalty_mask = min_distances < self.diversity_threshold
            penalty_factor = 1.0 - self.diversity_penalty_factor * (min_distances / self.diversity_threshold)
            penalty_factor = np.clip(penalty_factor, 0.1, 1.0)  # Ensure minimum value
            
            # Apply penalty to the acquisition values
            ei[penalty_mask] *= penalty_factor[penalty_mask]
        
        return ei.reshape(-1, 1)

    def _select_sampling_strategy(self):
        """Select sampling strategy using multi-armed bandit"""
        # Epsilon-greedy selection with small epsilon
        epsilon = 0.1
        
        if self.rng.random() < epsilon:
            # Explore: randomly select a strategy
            strategy_idx = self.rng.choice(3)
        else:
            # Exploit: select the best performing strategy
            strategy_idx = np.argmax(self.strategy_weights)
        
        return strategy_idx

    def _generate_candidates(self, n_candidates):
        """Generate candidates based on selected sampling strategy"""
        strategy_idx = self._select_sampling_strategy()
        
        if strategy_idx == 0:  # Random strategy
            # Generate random points
            candidates = self._sample_points(n_candidates)
            
        elif strategy_idx == 1:  # Local search around best point
            # Sample points around the current best point
            if self.X is not None and len(self.X) > 0:
                best_x = self.X[np.argmin(self.y)]
                candidates = []
                for _ in range(n_candidates):
                    # Generate point near best point with some randomness
                    candidate = best_x + (self.rng.random(self.dim) - 0.5) * 2.0
                    candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                    candidates.append(candidate)
                candidates = np.array(candidates)
            else:
                candidates = self._sample_points(n_candidates)
                
        else:  # Global strategy (Latin hypercube)
            candidates = self._sample_points(n_candidates)
            
        return candidates

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using population-based approach"""
        # Generate candidates from different strategies
        n_candidates = min(100 * self.dim, 500)  # Limit candidates for efficiency
        
        # Use the sampling strategy to generate diverse candidates
        candidates = self._generate_candidates(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._diversity_adjusted_ei(candidates)
        
        # Select top candidates using a more efficient approach - just take the best ones
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        # Update strategy rewards based on performance
        if len(self.best_y_history) > 1:
            improvement = self.best_y_history[-2] - self.best_y_history[-1]
            strategy_idx = self._select_sampling_strategy()
            self.strategy_counts[strategy_idx] += 1
            self.strategy_rewards[strategy_idx] += improvement
            
            # Update weights using softmax
            if np.sum(self.strategy_counts) > 0:
                avg_rewards = self.strategy_rewards / (self.strategy_counts + 1e-8)
                weights = np.exp(avg_rewards * 2.0)  # Temperature parameter
                self.strategy_weights = weights / np.sum(weights)
        
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
            
            # Early exit when budget is nearly exhausted
            if self.n_evals + 2 > self.budget:
                break
                
            # Select next points using acquisition function with diversity
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
