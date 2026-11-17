from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings

class EfficientAdaptiveTrustRegionBO:
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
        self.kernel_type = 'matern'  # Switch between 'matern' and 'rbf' based on problem
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 1.2
        self.progress_threshold = 0.005  # Reduced threshold for more sensitive progress detection
        self.stagnation_count = 0
        self.max_stagnation = 3  # Reduced from previous version for faster response
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.05  # Increased initial exploration weight
        self.exploration_decay = 0.98  # Faster decay of exploration
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for trust region adaptation
        self.best_y_history = []
        self.last_improvement_iter = 0
        self.iteration_count = 0
        
        # Model switching parameters
        self.model_switch_threshold = 0.1  # Threshold to switch between kernels
        self.kernel_switch_count = 0
        self.max_kernel_switches = 3

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling properties"""
        # Create Sobol sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model with adaptive kernel selection"""
        # Use a GP with ARD Matern kernel for smooth functions
        if self.kernel_type == 'matern':
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        else:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            
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

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function with dynamic exploration weight"""
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

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using multi-start optimization"""
        # Initialize with a good set of starting points within the trust region
        n_starting_points = min(30 * self.dim, 500)  # Reduced for efficiency
        
        # Generate candidates using hybrid approach: random + differential evolution
        candidates = []
        
        if self.X is not None and len(self.X) > 0:
            # Use the current best point as center of trust region
            best_x = self.X[np.argmin(self.y)]
            
            # Sample points within trust region (50% random, 50% deterministic)
            random_points = []
            while len(random_points) < n_starting_points // 2:
                # Generate random points in the trust region
                sample = best_x + (self.rng.random(self.dim) - 0.5) * 2 * self.trust_region_radius
                
                # Ensure points are within bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                random_points.append(sample)
            
            candidates.extend(random_points)
            
            # Add points from differential evolution optimization
            try:
                # Define the acquisition function for optimization
                def acq_func(x):
                    return -self._acquisition_function(x.reshape(1, -1))[0, 0]
                
                # Run differential evolution with multiple restarts
                de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
                de_result = differential_evolution(
                    acq_func,
                    de_bounds,
                    maxiter=20,
                    popsize=15,
                    seed=int(self.rng.integers(2**31 - 1)),
                    strategy='best1bin',
                    atol=1e-6
                )
                
                if not np.isnan(de_result.fun):
                    candidates.append(de_result.x)
            except:
                pass  # Fall back to random sampling if DE fails
            
        else:
            # If no points yet, use Sobol sampling
            candidates = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(np.array(candidates))
        
        # Select top candidates using a more efficient approach - just take the best ones
        top_indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = np.array(candidates)[top_indices]
        
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

    def _update_trust_region(self):
        """Update trust region based on recent progress with more responsive detection"""
        if len(self.best_y_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        current_best = self.best_y_history[-1]
        previous_best = self.best_y_history[-2]
        
        improvement = (previous_best - current_best) / abs(previous_best)
        
        # Update stagnation count
        if improvement > self.progress_threshold:
            self.stagnation_count = 0
            # Expand trust region if progress is good
            self.trust_region_radius = min(
                self.trust_region_radius * self.trust_region_expand_factor,
                self.max_trust_region_radius
            )
        else:
            self.stagnation_count += 1
            # Shrink trust region if no significant progress
            if self.stagnation_count >= self.max_stagnation:
                self.trust_region_radius = max(
                    self.trust_region_radius * self.trust_region_shrink_factor,
                    self.min_trust_region_radius
                )
                self.stagnation_count = 0

    def _check_restart_condition(self):
        """Check if we should restart the optimization"""
        # Restart if trust region becomes too small and no progress
        if (self.trust_region_radius <= self.min_trust_region_radius and 
            self.stagnation_count >= self.max_stagnation):
            return True
        return False

    def _adapt_exploration_weight(self):
        """Adapt exploration weight based on optimization progress"""
        # Reduce exploration weight over time to shift from exploration to exploitation
        if self.iteration_count > 0:
            self.exploration_weight *= self.exploration_decay
            
        # Ensure we don't go below a minimum exploration level
        self.exploration_weight = max(self.exploration_weight, 0.01)

    def _adapt_kernel(self):
        """Adapt kernel based on problem characteristics"""
        if len(self.y) < 10 or self.kernel_switch_count >= self.max_kernel_switches:
            return
            
        # Simple heuristic: switch to RBF if we're in a flat region (low variance)
        recent_y = self.y[-5:]
        if np.std(recent_y) < 0.1:
            # Switch to RBF kernel for potentially non-smooth functions
            if self.kernel_type == 'matern':
                self.kernel_type = 'rbf'
                self.kernel_switch_count += 1

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Sobol sequence design
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
            
            # Adapt exploration weight
            self._adapt_exploration_weight()
            
            # Adapt kernel if needed
            self._adapt_kernel()
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # Check for restart condition
            if self._check_restart_condition():
                # Reset trust region and continue
                self.trust_region_radius = 1.0
                self.stagnation_count = 0
            
            # Early exit when budget is nearly exhausted
            if self.n_evals + 2 > self.budget:
                break
                
            # Select next points using acquisition function
            next_points = self._select_next_points(self.batch_size)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
            self.iteration_count += 1
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
