from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from scipy.optimize import differential_evolution
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
            
        # Surrogate model parameters
        self.gp_models = []
        self.model_weights = []
        self.active_model_idx = 0
        
        # Acquisition function parameters
        self.acq_functions = ['ei', 'lcb', 'pi']
        self.acq_weights = np.array([1.0, 1.0, 1.0])  # weights for each acquisition function
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.1
        
        # Diversity tracking
        self.diversity_history = []
        self.diversity_threshold = 0.5  # threshold to determine when to change exploration strategy
        
        # Bandit selection parameters
        self.arm_rewards = np.zeros(len(self.acq_functions))
        self.arm_counts = np.zeros(len(self.acq_functions))
        self.epsilon = 0.1  # exploration parameter for bandit
        
        # Kernel configuration parameters
        self.kernels_config = [
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5),
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5),
            RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        ]
        
        # Progress tracking
        self.best_y_history = []
        self.last_improvement_iter = 0
        
        # Batch selection parameters
        self.batch_size = 1
        self.diversity_factor = 0.3

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_models(self, X, y):
        """Fit multiple GP models with different kernels"""
        self.gp_models = []
        for kernel in self.kernels_config:
            # Add noise to kernel
            kernel_with_noise = kernel + WhiteKernel(noise_level=self.noise_level, 
                                                   noise_level_bounds=(1e-10, 1e1))
            
            model = GaussianProcessRegressor(
                kernel=kernel_with_noise,
                alpha=0.0,
                n_restarts_optimizer=2,
                normalize_y=True,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            
            try:
                model.fit(X, y)
                self.gp_models.append(model)
            except Exception:
                # If fitting fails, create a simple model with default kernel
                simple_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=self.noise_level)
                simple_model = GaussianProcessRegressor(
                    kernel=simple_kernel,
                    alpha=0.0,
                    n_restarts_optimizer=0,
                    normalize_y=True,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                simple_model.fit(X, y)
                self.gp_models.append(simple_model)

    def _acquisition_function(self, X, acq_type='ei'):
        """Expected Improvement acquisition function"""
        if len(self.gp_models) == 0:
            return np.zeros((X.shape[0], 1))
            
        # Select current model
        model = self.gp_models[self.active_model_idx]
        
        # Get predictions from GP
        mu, sigma = model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        if acq_type == 'ei':
            # Expected Improvement
            improvement = y_min - mu
            from scipy.stats import norm
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return ei.reshape(-1, 1)
        elif acq_type == 'lcb':
            # Lower Confidence Bound
            lcb = mu - self.exploration_weight * sigma
            return (-lcb).reshape(-1, 1)  # negative because we minimize
        elif acq_type == 'pi':
            # Probability of Improvement
            improvement = y_min - mu
            from scipy.stats import norm
            z = improvement / sigma
            pi = norm.cdf(z)
            return pi.reshape(-1, 1)
        else:
            return np.zeros((X.shape[0], 1))

    def _calculate_diversity(self):
        """Calculate diversity of current population"""
        if self.X is None or len(self.X) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = cdist(self.X, self.X)
        # Set diagonal to zero (distance to itself)
        np.fill_diagonal(distances, 0)
        
        if distances.size == 0:
            return 0.0
            
        # Return mean of all non-diagonal elements
        return np.mean(distances[distances > 0])

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-armed bandit approach"""
        # First, calculate current diversity
        current_diversity = self._calculate_diversity()
        self.diversity_history.append(current_diversity)
        
        # Select acquisition function based on Thompson sampling or epsilon-greedy
        if self.rng.random() < self.epsilon or len(self.arm_counts) == 0:
            # Exploration: randomly select an arm
            selected_arm = self.rng.integers(len(self.acq_functions))
        else:
            # Exploitation: select best arm based on rewards
            # Normalize rewards by counts (avoid division by zero)
            normalized_rewards = np.zeros_like(self.arm_rewards)
            mask = self.arm_counts > 0
            if np.any(mask):
                normalized_rewards[mask] = self.arm_rewards[mask] / self.arm_counts[mask]
            
            # Add exploration bonus (Thompson sampling style)
            exploration_bonus = np.sqrt(2 * np.log(np.sum(self.arm_counts) + 1) / (self.arm_counts + 1e-8))
            if len(self.arm_counts) > 0:
                normalized_rewards += exploration_bonus
            
            selected_arm = np.argmax(normalized_rewards)
        
        # Generate candidates using the selected acquisition function
        n_candidates = min(200 * self.dim, 1000)
        
        # Use a combination of random sampling and local search
        candidates = []
        
        # Add some random points for exploration
        random_points = self._sample_points(n_candidates // 2)
        candidates.extend(random_points)
        
        # Add points near current best
        if self.X is not None and len(self.X) > 0:
            best_x = self.X[np.argmin(self.y)]
            # Generate points around the best point with some randomness
            for _ in range(n_candidates // 2):
                # Add small random perturbation
                candidate = best_x + (self.rng.random(self.dim) - 0.5) * 0.5
                # Clip to bounds
                candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates, self.acq_functions[selected_arm])
        
        # Select top candidates based on diversity
        if len(candidates) > batch_size:
            # Calculate diversity scores for candidates
            diversity_scores = []
            for i, candidate in enumerate(candidates):
                # Calculate how diverse this candidate is compared to existing points
                if self.X is not None and len(self.X) > 0:
                    distances = cdist([candidate], self.X)[0]
                    min_distance = np.min(distances)
                    diversity_score = 1.0 / (min_distance + 1e-8)
                else:
                    diversity_score = 1.0
                diversity_scores.append(diversity_score)
            
            diversity_scores = np.array(diversity_scores)
            
            # Combine acquisition value with diversity score
            combined_score = acq_values.flatten() * (1.0 - self.diversity_factor) + \
                           diversity_scores * self.diversity_factor
            
            # Select best candidates
            top_indices = np.argsort(combined_score)[-batch_size:]
            selected_points = candidates[top_indices]
        else:
            # If not enough candidates, just return the top ones
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

    def _update_bandit_rewards(self, acq_type_idx, reward):
        """Update rewards for the selected arm"""
        self.arm_rewards[acq_type_idx] += reward
        self.arm_counts[acq_type_idx] += 1

    def _switch_model_if_needed(self):
        """Switch to a different model if performance degrades"""
        if len(self.gp_models) <= 1:
            return
            
        # Simple check: if the last few evaluations didn't improve significantly, switch models
        if len(self.best_y_history) >= 5:
            recent_improvement = self.best_y_history[-5] - self.best_y_history[-1]
            if recent_improvement < 1e-6:
                # No improvement for a while, try a different model
                self.active_model_idx = (self.active_model_idx + 1) % len(self.gp_models)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP models
        self._fit_gp_models(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self._fit_gp_models(self.X, self.y)
            
            # Switch model if needed
            self._switch_model_if_needed()
            
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            current_best = np.min(self.y)
            self.best_y_history.append(current_best)
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
