# Description
**EfficientGlobalOptimizationBO**: This Bayesian Optimization algorithm uses an efficient global optimization approach with a sophisticated initial design, robust Gaussian Process surrogate model, and a novel acquisition function optimizer based on a small evolution strategy. It employs expected improvement acquisition function with a trust region mechanism to balance exploration and exploitation.

# Justification 
The key innovation in this algorithm is replacing the multi-start L-BFGS-B acquisition function optimizer with a small evolution strategy (ES) for global optimization. This change offers several advantages:

1. **Robustness**: Unlike gradient-based optimizers like L-BFGS-B, ES is derivative-free and less sensitive to local optima, which is crucial for the challenging Rosenbrock function.

2. **Efficiency**: The small ES uses only 10-20 individuals, making it computationally efficient even with limited budget constraints. This is important because we have a budget of 600 evaluations, and each acquisition function optimization step should be fast.

3. **Parallelism**: ES naturally allows for parallel evaluation of candidate solutions, which can be beneficial when multiple acquisitions are needed.

4. **Global search capability**: The evolutionary approach helps escape poor local optima that L-BFGS-B might get stuck in during acquisition function optimization, especially in the highly non-convex Rosenbrock landscape.

5. **Trust region adaptation**: The algorithm maintains a trust region mechanism that adapts dynamically based on progress, allowing efficient exploration early on and focused exploitation near the optimum.

The Gaussian Process surrogate model with ARD Matern kernel is maintained as it provides good flexibility for modeling smooth functions while being computationally efficient. The expected improvement acquisition function remains appropriate for balancing exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.spatial.distance import cdist
import warnings

class EfficientGlobalOptimizationBO:
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
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 1.2
        self.progress_threshold = 0.01  # Minimum improvement to consider as progress
        self.stagnation_count = 0
        self.max_stagnation = 5  # Number of iterations without progress before shrinking
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploration_weight = 0.01
        
        # For batch selection
        self.batch_size = 1
        
        # Performance tracking for trust region adaptation
        self.best_y_history = []
        self.last_improvement_iter = 0

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
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
            n_restarts_optimizer=3,  # Increased restarts for better optimization
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function"""
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

    def _evolution_strategy_optimizer(self, n_candidates=20):
        """Optimize acquisition function using small evolution strategy"""
        # Initialize population with random points within bounds
        population = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for initial population
        fitness_values = self._acquisition_function(population).flatten()
        
        # Evolutionary optimization loop
        max_generations = 10
        for generation in range(max_generations):
            # Sort population by fitness (descending order)
            sorted_indices = np.argsort(fitness_values)[::-1]
            population = population[sorted_indices]
            fitness_values = fitness_values[sorted_indices]
            
            # Select top performers (elitism)
            elite_size = max(1, n_candidates // 4)
            elite = population[:elite_size]
            
            # Generate offspring through mutation
            offspring = []
            for i in range(n_candidates - elite_size):
                # Select parent from elite
                parent_idx = self.rng.integers(elite_size)
                parent = elite[parent_idx]
                
                # Create offspring by adding Gaussian noise
                noise = self.rng.normal(0, self.trust_region_radius / 4, self.dim)
                child = parent + noise
                
                # Ensure bounds are respected
                child = np.clip(child, self.bounds[0], self.bounds[1])
                offspring.append(child)
            
            # Add elite to offspring
            offspring = np.vstack([offspring, elite])
            
            # Evaluate offspring
            offspring_fitness = self._acquisition_function(offspring).flatten()
            
            # Replace population with best individuals
            all_individuals = np.vstack([population, offspring])
            all_fitness = np.concatenate([fitness_values, offspring_fitness])
            
            # Sort and keep best individuals
            sorted_indices = np.argsort(all_fitness)[::-1]
            population = all_individuals[sorted_indices[:n_candidates]]
            fitness_values = all_fitness[sorted_indices[:n_candidates]]
        
        # Return the best individual found
        best_idx = np.argmax(fitness_values)
        return population[best_idx].reshape(1, -1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using evolution strategy optimizer"""
        # Use small evolution strategy for acquisition function optimization
        n_candidates = min(max(20, 5 * self.dim), 100)
        
        # Generate candidates and optimize acquisition function
        best_candidate = self._evolution_strategy_optimizer(n_candidates)
        
        return best_candidate

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
        """Update trust region based on recent progress"""
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
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # Check for restart condition
            if self._check_restart_condition():
                # Reset trust region and continue
                self.trust_region_radius = 1.0
                self.stagnation_count = 0
            
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
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Feedback
 The algorithm EfficientGlobalOptimizationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3431 with standard deviation 0.0091.

took 29.08 seconds to run.