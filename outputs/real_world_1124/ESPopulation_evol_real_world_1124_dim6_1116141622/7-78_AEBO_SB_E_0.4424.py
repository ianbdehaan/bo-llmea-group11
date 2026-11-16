from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")

class AEBO_SB_E:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, dim * 2))  # Adaptive initial points based on dimensionality

        # Do not add any other arguments without a default value
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.model = None
        self.is_initial_phase = True
        self.acquisition_batch_size = 1  # Start with small batches
        self.last_model_update = 0
        self.model_retrain_interval = 5
        self.ei_threshold = 0.1  # Threshold for switching to exploitation
        self.exploitation_switch_budget_ratio = 0.35  # When to switch to exploitation phase
        self.performance_history = []
        self.min_improvement_threshold = 1e-4  # Minimum improvement needed to justify model retraining
        self.acquisition_function_calls = 0  # Track acquisition function calls for debugging
        self.uncertainty_scaling_factor = 1.0  # Dynamic uncertainty scaling factor

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for diverse initial exploration
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit surrogate model with fixed hyperparameters for efficiency
        # Use a sparse GP with Matern kernel and fixed length scales to reduce optimization cost
        
        # Create a base kernel with fixed length scales to reduce optimization cost
        # Using a simple approach: fix the length scale to 1.0 and optimize only signal variance and noise
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1.0, 1.0), nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        # For computational efficiency, use a subset of points for fitting (especially important when budget is small)
        max_points_for_fit = min(100, X.shape[0])
        fit_indices = self.rng.choice(X.shape[0], size=max_points_for_fit, replace=False)
        X_fit = X[fit_indices]
        y_fit = y[fit_indices]
        
        # Fit the GP model with fewer restarts for efficiency
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=0,  # No restarts for even more efficiency
            random_state=int(self.rng.integers(2**31 - 1)),
            normalize_y=True
        )
        
        gp.fit(X_fit, y_fit)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function using Expected Improvement with enhanced uncertainty scaling
        # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
        
        if self.model is None:
            # If no model yet, return random acquisition values for exploration
            return np.ones((X.shape[0], 1)) * self.rng.random(X.shape[0]).reshape(-1, 1)
            
        # Get predictions from the model
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Find current best value (excluding noise)
        if self.y is not None and len(self.y) > 0:
            f_best = np.min(self.y)
        else:
            f_best = 0.0
            
        # Calculate EI using the formula
        improvement = f_best - mu
        
        # Add numerical stability
        sigma_safe = np.maximum(sigma, 1e-8)
        
        with np.errstate(divide='warn', invalid='ignore'):
            Z = improvement / sigma_safe
            ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
            
        # Clip negative values to zero
        ei = np.maximum(ei, 0.0)
        
        # Enhanced uncertainty scaling for early phase exploration
        if self.is_initial_phase:
            # Dynamic scaling that decreases with evaluations
            scale_factor = 1.0 + 0.5 * np.exp(-self.n_evals / max(1, self.budget * 0.3))
            ei = ei * scale_factor
            
        # Additional uncertainty-aware scaling to prevent over-exploration in later stages
        if not self.is_initial_phase:
            # Scale based on relative standard deviation
            mean_sigma = np.mean(sigma)
            if mean_sigma > 0:
                uncertainty_ratio = np.maximum(0.1, mean_sigma / (np.abs(f_best) + 1e-6))
                # Reduce EI when uncertainty is high (to encourage exploitation)
                ei = ei * np.exp(-uncertainty_ratio)
                
        self.acquisition_function_calls += 1
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using an improved hybrid approach:
        # - Use diverse sampling strategies for better exploration in early phases
        # - Switch to acquisition function optimization for exploitation
        
        if self.is_initial_phase or self.n_evals < 10:
            # Use diversified sampling during initial phase
            return self._sample_points(batch_size)
        
        # For more advanced selection, we'll optimize the acquisition function
        # Use a gradient-free approach with multiple restarts and better candidate generation
        
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple starting points for optimization
        n_restarts = min(5, max(2, self.budget // 30))
        
        # Generate candidates using multiple strategies to improve diversity
        candidates_list = []
        
        # Strategy 1: Random sampling around current best (with noise)
        if len(self.X) > 0:
            # Sample around the best point found so far with more sophisticated noise
            best_idx = np.argmin(self.y)
            best_x_found = self.X[best_idx]
            
            # Add different types of noise for better exploration
            noise_types = [
                self.rng.normal(0, 0.2, self.dim) * (self.bounds[1] - self.bounds[0]),  # Larger noise
                self.rng.uniform(-0.1, 0.1, self.dim) * (self.bounds[1] - self.bounds[0]),  # Uniform noise
                np.zeros(self.dim)  # No noise (exploitation)
            ]
            
            for noise in noise_types:
                candidate = np.clip(best_x_found + noise, self.bounds[0], self.bounds[1])
                candidates_list.append(candidate)
        
        # Strategy 2: Random sampling with better distribution
        random_candidates = self._sample_points(15)
        candidates_list.extend(random_candidates)
        
        # Strategy 3: LHS sampling with scrambling
        lhs_candidates = self._sample_points(8)
        candidates_list.extend(lhs_candidates)
        
        # Strategy 4: Quasi-Monte Carlo sampling for better coverage
        qmc_sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        qmc_candidates = qmc_sampler.random(n=5)
        qmc_candidates = self.bounds[0] + qmc_candidates * (self.bounds[1] - self.bounds[0])
        candidates_list.extend(qmc_candidates)
        
        # Strategy 5: Grid sampling for structured exploration
        if self.dim <= 4:  # Only for low dimensions
            grid_points = self._generate_grid_points()
            candidates_list.extend(grid_points)
        
        candidates = np.array(candidates_list)
        
        # Evaluate acquisition values for all candidates
        acq_vals = self._acquisition_function(candidates)
        
        # Select the best candidate from these samples using a more robust approach
        if len(acq_vals) > 0:
            # Use top-k selection with some randomness to avoid getting stuck in local maxima
            k = min(20, len(acq_vals))
            top_k_indices = np.argpartition(-acq_vals.flatten(), min(k, len(acq_vals)))[:k]
            best_candidate_idx = self.rng.choice(top_k_indices)
            best_x = candidates[best_candidate_idx].reshape(1, -1)
        
        # If no good candidates found, fall back to random
        if best_x is None:
            return self._sample_points(batch_size)
            
        return best_x

    def _generate_grid_points(self):
        """Generate a structured grid for low-dimensional problems"""
        n_points_per_dim = 3
        grid_1d = np.linspace(self.bounds[0, 0], self.bounds[1, 0], n_points_per_dim)
        if self.dim == 1:
            return grid_1d.reshape(-1, 1)
        elif self.dim == 2:
            x1, x2 = np.meshgrid(grid_1d, grid_1d, indexing='ij')
            return np.column_stack([x1.ravel(), x2.ravel()])
        else:
            # For higher dimensions, use a sparse grid approach
            points = []
            for i in range(n_points_per_dim ** self.dim):
                coords = []
                temp_i = i
                for d in range(self.dim):
                    coords.append(temp_i % n_points_per_dim)
                    temp_i //= n_points_per_dim
                point = [grid_1d[c] for c in coords]
                points.append(point)
            return np.array(points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        if X is None or len(X) == 0:
            return np.array([]).reshape(0, 1)
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except Exception as e:
                # If function evaluation fails, use a large value to penalize
                print(f"Function evaluation failed: {e}")
                y_vals.append(1e6)
                
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _should_retrain_model(self):
        # Decide whether to retrain the model based on performance gains
        if self.n_evals < 10:
            return True
        
        # Retrain periodically or when enough new data is available
        if (self.n_evals - self.last_model_update) >= self.model_retrain_interval:
            return True
            
        # Also retrain if we have enough points for a good model fit and significant improvement
        if len(self.X) > 20:
            # Calculate recent improvement in performance
            if len(self.performance_history) >= 3:
                recent_performance = np.mean(self.performance_history[-3:])
                current_performance = np.min(self.y)
                improvement = abs(recent_performance - current_performance)
                
                # Only retrain if there's a significant improvement
                if improvement > self.min_improvement_threshold:
                    return True
            
            # Occasionally retrain even without significant improvement to prevent stagnation
            if np.random.rand() < 0.25:
                return True
                
        return False

    def _calculate_smart_batch_size(self):
        """Calculate batch size based on remaining budget and optimization state"""
        remaining_budget = self.budget - self.n_evals
        
        # If we're close to the end, use single evaluation
        if remaining_budget <= 5:
            return 1
            
        # If we're in early phase, use larger batches for efficient exploration
        if self.n_evals < self.budget * 0.2:
            base_batch = min(4, max(1, remaining_budget // 15))
            return min(base_batch, remaining_budget)
            
        # If we're in mid-phase, use moderate batch sizes
        elif self.n_evals < self.budget * 0.7:
            base_batch = min(3, max(1, remaining_budget // 12))
            return min(base_batch, remaining_budget)
            
        # In late phase, reduce batch size for focused exploitation
        else:
            return 1

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Fit the initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Calculate dynamic batch size based on remaining budget and optimization state
            batch_size = self._calculate_smart_batch_size()
            
            # Check if we should switch to exploitation phase based on both budget and EI values
            if (self.n_evals > self.budget * self.exploitation_switch_budget_ratio) or \
               (len(self.y) > 10 and np.mean(self.y[-5:]) < np.min(self.y) + self.ei_threshold):
                self.is_initial_phase = False
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_values)
            
            # Track performance for model retraining decisions
            if len(self.y) > 0:
                self.performance_history.append(np.min(self.y))
            
            # Refit model periodically when beneficial
            if self._should_retrain_model():
                try:
                    self.model = self._fit_model(self.X, self.y)
                    self.last_model_update = self.n_evals
                except Exception as e:
                    print(f"Model fitting failed: {e}")
            
            # Check if we've reached the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
