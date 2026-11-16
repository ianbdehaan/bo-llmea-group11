from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings

class AdaptiveLocalSearchBO:
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
            
        # Surrogate models parameters
        self.gp_model = None
        self.rf_model = None
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        self.diversity_penalty = 0.1
        
        # Local search parameters
        self.local_search_threshold = 0.05  # Threshold for switching to local search
        self.is_local_phase = False
        self.local_search_radius = 1.0
        self.min_local_radius = 0.01
        self.max_local_radius = 2.0
        
        # Performance tracking
        self.best_y_history = []
        self.last_improvement_iter = 0
        self.improvement_count = 0

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling with maximin property
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_surrogate_models(self, X, y):
        """Fit both GP and RF surrogate models"""
        # Fit Gaussian Process model
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp_model.fit(X, y)
        
        # Fit Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=int(self.rng.integers(2**31 - 1)),
            bootstrap=True
        )
        
        rf_model.fit(X, y.flatten())
        
        return gp_model, rf_model

    def _ensemble_predict(self, X):
        """Get predictions from both models and combine them"""
        if self.gp_model is None or self.rf_model is None:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
            
        # Get predictions from GP
        gp_mean, gp_std = self.gp_model.predict(X, return_std=True)
        
        # Get predictions from RF
        rf_pred = self.rf_model.predict(X)
        
        # Combine predictions - weighted average with GP having more weight
        # for better uncertainty estimation
        mean = 0.7 * gp_mean + 0.3 * rf_pred
        std = np.sqrt(0.7 * (gp_std ** 2) + 0.3 * (np.std(rf_pred) ** 2))
        
        return mean.reshape(-1, 1), std.reshape(-1, 1)

    def _acquisition_function(self, X):
        """Expected Improvement with diversity penalty acquisition function"""
        if self.gp_model is None or self.rf_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from ensemble
        mu, sigma = self._ensemble_predict(X)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add diversity penalty to encourage exploration of different regions
        if self.X is not None and len(self.X) > 0:
            # Calculate pairwise distances from new candidates to existing points
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            
            # Diversity penalty - penalize points that are too close to existing points
            diversity_penalty = np.exp(-self.diversity_penalty * min_distances)
            ei = ei * diversity_penalty.reshape(-1, 1)
        
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Determine search strategy based on current state
        if self.is_local_phase:
            # Use local search around best point
            n_starting_points = min(50 * self.dim, 200)
            
            if self.X is not None and len(self.X) > 0:
                best_x = self.X[np.argmin(self.y)]
                
                # Sample points within local radius
                candidates = []
                while len(candidates) < n_starting_points:
                    sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.local_search_radius
                    sample = np.clip(sample, self.bounds[0], self.bounds[1])
                    candidates.append(sample)
                
                candidates = np.array(candidates)
            else:
                candidates = self._sample_points(n_starting_points)
        else:
            # Use global search with maximin Latin hypercube sampling
            n_starting_points = min(100 * self.dim, 500)
            candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates using a simple heuristic
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
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

    def _update_local_search_strategy(self):
        """Dynamically update local search strategy based on optimization progress"""
        if len(self.best_y_history) < 2:
            return
            
        # Calculate recent improvement
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        # Switch to local search if improvement is small but we have some progress
        if improvement < self.local_search_threshold and len(self.best_y_history) > 5:
            self.is_local_phase = True
            # Increase local search radius for more thorough exploration
            self.local_search_radius = min(
                self.local_search_radius * 1.2,
                self.max_local_radius
            )
        elif improvement > 0.02:  # If we're making good progress, stay global
            self.is_local_phase = False
            self.local_search_radius = max(
                self.local_search_radius * 0.8,
                self.min_local_radius
            )

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial surrogate models
        self.gp_model, self.rf_model = self._fit_surrogate_models(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self.gp_model, self.rf_model = self._fit_surrogate_models(self.X, self.y)
            
            # Update local search strategy based on progress
            self._update_local_search_strategy()
            
            # Select next points using acquisition function
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
