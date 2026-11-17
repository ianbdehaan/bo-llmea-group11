from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution
import warnings

class EfficientRobustBOAO:
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
            
        # Acquisition function parameters
        self.noise_level = 1e-6
        
        # Model parameters
        self.model = None
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model with reduced computational cost"""
        # Use a sparse GP with ARD Matern kernel for robustness in high dimensions
        # Matern kernel with nu=2.5 provides good balance between smoothness and flexibility
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with fewer restarts for efficiency
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=1,  # Reduced from 3 to 1 for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _multi_objective_acquisition(self, X):
        """Multi-objective acquisition function combining expected improvement with local exploration"""
        if self.model is None:
            return np.zeros((X.shape[0], 2))  # Return zeros for both objectives
            
        # Get predictions from the GP model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Objective 1: Expected Improvement (EI)
        progress_ratio = self.n_evals / self.budget
        
        # Adaptively adjust exploration based on budget progress
        if progress_ratio < 0.3:
            xi = 0.01 * (1 - progress_ratio) + 0.001  # More exploration early
        elif progress_ratio < 0.6:
            xi = 0.005  # Moderate exploration
        else:
            xi = 0.001  # Less exploration later
            
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        improvement = y_min - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Objective 2: Uncertainty (inverse of standard deviation)
        uncertainty = 1.0 / (sigma + 1e-8)
        
        # Normalize objectives to similar scale
        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-8)
        uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)
        
        # Return both objectives
        return np.column_stack([ei_normalized, uncertainty_normalized])

    def _select_next_points(self):
        """Select next points using adaptive multi-objective optimization approach"""
        # Use a hybrid approach: random sampling with local refinement and adaptive restarts
        
        # For small budgets, use more diverse sampling
        if self.n_evals < self.budget * 0.2:
            pop_size = min(50, 10 * self.dim)
            n_local_refinements = max(1, pop_size // 10)
        else:
            pop_size = min(30, 5 * self.dim)
            n_local_refinements = max(1, pop_size // 5)
        
        # Generate initial population
        if self.X is not None and len(self.X) > 0:
            # Use best points as starting locations with some diversity
            best_indices = np.argsort(self.y.flatten())[:min(3, len(self.y))]
            start_points = self.X[best_indices]
            
            # Generate diverse initial population
            population = []
            for i in range(pop_size):
                if i < len(start_points):
                    point = start_points[i % len(start_points)].copy()
                else:
                    point = self._sample_points(1)[0]
                
                # Add noise for diversity (more at early stages)
                if self.n_evals < self.budget * 0.3:
                    noise_scale = 0.2 * (self.bounds[1] - self.bounds[0])
                else:
                    noise_scale = 0.1 * (self.bounds[1] - self.bounds[0])
                    
                noise = self.rng.normal(0, noise_scale)
                point += noise
                
                # Clip to bounds
                point = np.clip(point, self.bounds[0], self.bounds[1])
                population.append(point)
        else:
            # No previous points, generate random population
            population = self._sample_points(pop_size)
        
        # Evaluate multi-objective acquisition function for the entire population
        population_array = np.array(population)
        acq_values = self._multi_objective_acquisition(population_array)
        
        # Use adaptive weighted sum approach to select best points
        progress_ratio = self.n_evals / self.budget
        
        # Dynamically adjust weights based on budget progress
        if progress_ratio < 0.3:
            # Early stage: prioritize EI (exploitation) more heavily
            weights = np.array([0.8, 0.2])
        elif progress_ratio < 0.6:
            # Middle stage: balanced approach
            weights = np.array([0.5, 0.5])
        else:
            # Later stage: prioritize uncertainty (exploration) more heavily
            weights = np.array([0.2, 0.8])
            
        weighted_scores = np.sum(acq_values * weights, axis=1)
        
        # Select top candidates based on weighted scores
        sorted_indices = np.argsort(weighted_scores)[::-1]
        selected_points = np.array([population[i] for i in sorted_indices[:self.batch_size]])
        
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
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Select next points using acquisition function
            next_points = self._select_next_points()
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

# Import norm for the acquisition function
from scipy.stats import norm
