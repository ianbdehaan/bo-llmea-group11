from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import warnings

class EfficientRobustBOMO:
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
        """Fit and tune surrogate model using Random Forest"""
        # Use Random Forest with adaptive parameters for robustness
        # The number of trees is set to balance accuracy and computational cost
        n_estimators = min(100, max(10, 2 * self.dim))
        
        # Use a relatively shallow tree depth to avoid overfitting
        max_depth = min(10, max(3, self.dim // 2))
        
        # Random Forest with bootstrap sampling and variance estimation
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=3,
            min_samples_leaf=2,
            bootstrap=True,
            oob_score=False,  # Not using OOB for efficiency
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single thread to ensure reproducibility
        )
        
        model.fit(X, y.flatten())
        return model

    def _multi_objective_acquisition(self, X):
        """Multi-objective acquisition function combining expected improvement with local exploration"""
        if self.model is None:
            return np.zeros((X.shape[0], 2))  # Return zeros for both objectives
            
        # Get predictions from the RF model
        mu = self.model.predict(X)
        
        # Get variance estimates for uncertainty quantification
        # For Random Forest, we need to compute this manually or use bootstrap samples
        # Since RF doesn't provide direct variance, we'll approximate it using tree variances
        
        # Simple approach: use the average variance across trees as proxy for uncertainty
        if hasattr(self.model, 'estimators_'):
            # Calculate variance of predictions across all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            sigma_squared = np.var(tree_predictions, axis=0)
            sigma = np.sqrt(sigma_squared)
        else:
            # Fallback: use a constant uncertainty estimate if variance not available
            sigma = np.ones(X.shape[0]) * 0.1
            
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Objective 1: Expected Improvement (EI)
        progress_ratio = self.n_evals / self.budget
        # Decrease exploration as we progress through budget
        xi = 0.01 * (1 - progress_ratio) + 0.001
        
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
        """Select next points using multi-objective optimization approach"""
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
        
        # Use weighted sum approach to select best points (simple but effective)
        # Weight towards EI in early stages, uncertainty in later stages
        progress_ratio = self.n_evals / self.budget
        
        # Early stage: prioritize EI
        if progress_ratio < 0.3:
            weights = np.array([0.7, 0.3])
        elif progress_ratio < 0.6:
            weights = np.array([0.5, 0.5])
        else:
            weights = np.array([0.3, 0.7])  # Later stage: prioritize uncertainty
            
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
