from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

class ABO_HS_EF:
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
        self.early_explore_factor = 1.0  # Factor to increase exploration in early stages

    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for diverse initial exploration
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Use a sparse GP with Matern kernel for efficiency
        # Reduce computational cost by using subset of data for fitting
        
        # Create a base kernel (Matern with automatic relevance detection)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        # For computational efficiency, use a subset of points for fitting (especially important when budget is small)
        max_points_for_fit = min(100, X.shape[0])
        if X.shape[0] > 50:
            fit_indices = self.rng.choice(X.shape[0], size=max_points_for_fit, replace=False)
        else:
            fit_indices = np.arange(X.shape[0])
        X_fit = X[fit_indices]
        y_fit = y[fit_indices]
        
        # Fit the GP model with limited optimization restarts for efficiency
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=1,  # Reduced number of restarts for efficiency
            random_state=int(self.rng.integers(2**31 - 1)),
            normalize_y=True
        )
        
        gp.fit(X_fit, y_fit)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function using Expected Improvement with uncertainty scaling
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
        with np.errstate(divide='warn'):
            Z = improvement / (sigma + 1e-8)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        # Add uncertainty scaling factor to encourage exploration in early stages
        if self.is_initial_phase:
            # Increase acquisition values for exploration in early phase with adaptive factor
            self.early_explore_factor = 1.0 + 0.5 * np.exp(-self.n_evals / max(1, self.budget * 0.3))
            ei = ei * self.early_explore_factor
            
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using a hybrid approach:
        # - Use random search in early phase for exploration
        # - Switch to acquisition function optimization for exploitation
        
        if self.is_initial_phase or self.n_evals < 10:
            # Use random sampling during initial phase
            return self._sample_points(batch_size)
        
        # For more advanced selection, we'll optimize the acquisition function
        # Use a simple gradient-free approach with multiple restarts but with limited computational cost
        
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple starting points for optimization
        n_restarts = min(5, max(2, self.budget // 40))
        
        # Generate candidates using different strategies to improve diversity
        candidates_list = []
        
        # Strategy 1: Random sampling around current best
        if len(self.X) > 0:
            # Sample around the best point found so far
            best_idx = np.argmin(self.y)
            best_x_found = self.X[best_idx]
            # Add some noise to explore nearby points
            for _ in range(3):
                noise = self.rng.normal(0, 0.1, self.dim) * (self.bounds[1] - self.bounds[0])
                candidate = np.clip(best_x_found + noise, self.bounds[0], self.bounds[1])
                candidates_list.append(candidate)
        
        # Strategy 2: Random sampling
        random_candidates = self._sample_points(10)
        candidates_list.extend(random_candidates)
        
        # Strategy 3: LHS sampling
        lhs_candidates = self._sample_points(5)
        candidates_list.extend(lhs_candidates)
        
        candidates = np.array(candidates_list)
        
        # Evaluate acquisition values for all candidates (limited to 50 candidates for efficiency)
        if len(candidates) > 50:
            # Sample a subset for evaluation
            candidate_indices = self.rng.choice(len(candidates), size=50, replace=False)
            eval_candidates = candidates[candidate_indices]
            acq_vals = self._acquisition_function(eval_candidates)
            # Get the best from the sampled candidates
            max_idx = np.argmax(acq_vals.flatten())
            best_x = eval_candidates[max_idx].reshape(1, -1)
        else:
            # Evaluate all candidates
            acq_vals = self._acquisition_function(candidates)
            # Select the best candidate from these samples
            max_idx = np.argmax(acq_vals.flatten())
            best_x = candidates[max_idx].reshape(1, -1)
        
        # If no good candidates found, fall back to random
        if best_x is None:
            return self._sample_points(batch_size)
            
        return best_x

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
            # Check if we should switch to exploitation phase
            if self.n_evals > self.budget * 0.4:
                self.is_initial_phase = False
                
            # Determine batch size based on remaining budget and current evaluation count
            remaining_budget = self.budget - self.n_evals
            if remaining_budget < 5:
                batch_size = 1
            elif remaining_budget < 20:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, max(1, remaining_budget // 20))
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_values)
            
            # Refit model periodically (every 3 evaluations or when we have enough data)
            if self.n_evals % 3 == 0 and len(self.X) > 10:
                try:
                    self.model = self._fit_model(self.X, self.y)
                except Exception as e:
                    print(f"Model fitting failed: {e}")
            
            # Check if we've reached the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
