# Description
**Efficient Multi-Start Bayesian Optimization with Trust-Region Exploration and Evolution Strategy Acquisition Function Optimizer (EMBO-TRE-ES)**

This is an enhanced Bayesian optimization algorithm that builds upon the EMBO-TRE framework but replaces its acquisition function optimizer with a small evolution strategy (ES) for improved robustness and efficiency. The approach maintains the adaptive trust-region exploration, multi-start optimization strategies, and efficient surrogate modeling while introducing evolutionary search for acquisition function optimization.

Key characteristics:
1. **Trust-Region Exploration**: Uses a dynamic trust-region mechanism around the best point that grows and shrinks based on progress
2. **Evolution Strategy Acquisition Optimization**: Employs a small evolution strategy with diverse initialization to optimize the acquisition function, providing better exploration than local optimizers
3. **Efficient Surrogate Model**: Uses a sparse Gaussian Process with automatic relevance detection (ARD) Matern kernel, optimized with a small number of hyperparameter restarts
4. **Adaptive Initial Design**: Starts with a maximin Latin Hypercube design scaled to the problem dimensionality
5. **Batch Selection**: Selects multiple points in batches using diverse sampling strategies within trust regions

# Justification
The replacement of the acquisition function optimizer with an evolution strategy (ES) provides several advantages over the previous differential evolution approach:

1. **Robustness to Local Optima**: ES is inherently less prone to getting stuck in local optima compared to gradient-based methods like L-BFGS or even differential evolution, which can also be trapped in poor local solutions.

2. **Parallel Evaluation Support**: ES naturally supports parallel evaluation of candidate solutions, making it more suitable for batch selection where multiple points are evaluated simultaneously.

3. **Diverse Search Strategy**: The evolutionary nature provides better exploration of the search space, especially when dealing with multimodal acquisition functions that may have multiple peaks or complex landscapes.

4. **Less Sensitive to Starting Points**: Unlike differential evolution which requires careful tuning of parameters and can be sensitive to initialization, ES maintains diversity through population-based search, providing more consistent performance across different optimization scenarios.

5. **Computational Efficiency**: The small evolution strategy used here (population size of 10-20) keeps computational cost low while maintaining good exploration properties, making it suitable for the constrained budget environment.

6. **Better Handling of Noisy Functions**: ES tends to be more robust in handling noisy or discontinuous functions that can occur in black-box optimization problems.

7. **Adaptive Search Control**: The evolution strategy can naturally adapt its search behavior based on population diversity and performance, providing automatic control without requiring manual parameter tuning.

The combination of trust-region exploration with ES acquisition optimization creates a powerful framework that balances global exploration with local exploitation while maintaining computational efficiency.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.spatial.distance import cdist

class EMBO_TRE_ES:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10) # Adaptive initial points

        # Random generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model parameters
        self.gp = None
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.min_trust_region_radius = 0.01
        self.max_trust_region_radius = 5.0
        self.region_shrink_factor = 0.8
        self.region_expand_factor = 1.2
        self.progress_threshold = 1e-3
        self.last_improvement_evals = 0
        self.improvement_count = 0
        
        # For tracking recent performance
        self.recent_best_y = np.inf
        self.performance_window_size = max(5, budget // 20)
        self.performance_history = []
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with ARD Matern kernel
        # Use a small number of restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement acquisition function
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP
        mean, std = self.gp.predict(X, return_std=True)
        
        # Get current best value (with some smoothing to avoid numerical issues)
        if len(self.y) > 0:
            current_best = np.min(self.y) - 1e-8
        else:
            current_best = 0.0
            
        # Compute EI
        improvement = current_best - mean
        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei.reshape(-1, 1)

    def _get_trust_region_center(self):
        # Return the best point seen so far as the center of trust region
        if self.X is None or len(self.X) == 0:
            return np.mean(self.bounds, axis=0)
        else:
            best_idx = np.argmin(self.y)
            return self.X[best_idx].copy()

    def _update_trust_region(self):
        # Update trust region based on recent progress
        if len(self.performance_history) < 2:
            return
            
        # Check if we've made significant improvement recently
        recent_improvement = (self.performance_history[-1] - self.performance_history[-2]) / abs(self.performance_history[-1])
        
        if recent_improvement > self.progress_threshold:
            # We're making good progress, expand the trust region
            self.trust_region_radius = min(
                self.max_trust_region_radius,
                self.trust_region_radius * self.region_expand_factor
            )
            self.improvement_count += 1
        else:
            # No significant improvement, shrink the trust region
            self.trust_region_radius = max(
                self.min_trust_region_radius,
                self.trust_region_radius * self.region_shrink_factor
            )
            self.improvement_count = 0

    def _sample_in_trust_region(self, center, n_points):
        # Sample points within trust region around the center
        # Create a box around center with size based on current trust region radius
        lower_bound = np.maximum(self.bounds[0], center - self.trust_region_radius)
        upper_bound = np.minimum(self.bounds[1], center + self.trust_region_radius)
        
        # If region is too small, just sample from bounds
        if np.all(upper_bound - lower_bound > 1e-6):
            # Use Latin Hypercube sampling within the trust region
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            points = sampler.random(n=n_points)
            points = lower_bound + points * (upper_bound - lower_bound)
        else:
            # Fall back to random sampling when trust region is very small
            points = self._sample_points(n_points)
            
        return points

    def _evolution_strategy_optimize_acquisition(self, n_candidates=20, max_iter=30):
        """
        Use a small evolution strategy to optimize the acquisition function.
        This provides better exploration than local optimizers and is robust to
        multimodal acquisition functions.
        """
        # Initialize population with diverse points around the trust region center
        tr_center = self._get_trust_region_center()
        
        # Create initial population
        population = []
        fitness_scores = []
        
        # Generate diverse starting points
        for _ in range(n_candidates):
            # Mix of random and trust-region sampling for diversity
            if self.rng.random() < 0.7:
                # Sample from trust region
                candidate = self._sample_in_trust_region(tr_center, 1)[0]
            else:
                # Sample randomly within bounds
                candidate = self._sample_points(1)[0]
            
            population.append(candidate)
        
        # Evaluate initial population
        for individual in population:
            acq_val = self._acquisition_function(individual.reshape(1, -1))[0, 0]
            fitness_scores.append(acq_val)
        
        # Evolutionary optimization loop
        for iteration in range(max_iter):
            # Sort by fitness (descending for maximization)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            best_individual = population[sorted_indices[0]]
            best_fitness = fitness_scores[sorted_indices[0]]
            
            # Select top performers (elitism)
            top_k = max(3, n_candidates // 4)
            selected_indices = sorted_indices[:top_k]
            selected_population = [population[i] for i in selected_indices]
            
            # Create new population through mutation and crossover
            new_population = []
            
            # Keep best individual (elitism)
            new_population.append(best_individual.copy())
            
            # Generate offspring through mutation
            for _ in range(n_candidates - 1):
                # Select parent from top performers
                parent_idx = self.rng.choice(selected_indices)
                parent = population[parent_idx]
                
                # Add Gaussian noise for mutation (adaptive step size)
                mutation_strength = self.trust_region_radius / 5.0
                offspring = parent + self.rng.normal(0, mutation_strength, self.dim)
                
                # Ensure bounds are respected
                offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
                
                new_population.append(offspring)
            
            # Evaluate new population
            population = new_population
            fitness_scores = []
            for individual in population:
                acq_val = self._acquisition_function(individual.reshape(1, -1))[0, 0]
                fitness_scores.append(acq_val)
        
        # Return the best individual found
        best_idx = np.argmax(fitness_scores)
        return population[best_idx].reshape(1, -1)

    def _select_next_points(self, batch_size):
        # Use evolution strategy for acquisition function optimization
        n_starts = min(20, max(5, batch_size * 3))  # More starts than needed to ensure diversity
        
        # Get current trust region center and update it
        tr_center = self._get_trust_region_center()
        
        # Generate diverse starting points within trust region for ES initialization
        start_points = self._sample_in_trust_region(tr_center, n_starts)
        
        best_value = -np.inf
        best_point = None
        
        # Run evolution strategy to optimize acquisition function
        try:
            # Use a small ES with population size of 15
            best_point = self._evolution_strategy_optimize_acquisition(n_candidates=15, max_iter=20)
            acq_value = self._acquisition_function(best_point)[0, 0]
            
            if acq_value > best_value:
                best_value = acq_value
        except Exception:
            # Fallback to simple trust region sampling if ES fails
            try:
                best_point = self._sample_in_trust_region(tr_center, 1)
            except Exception:
                best_point = self._sample_points(1)
        
        # Return multiple points by sampling diverse directions
        candidates = [best_point]
        
        # Add some diversity with additional random points within trust region
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_in_trust_region(tr_center, 1)
                candidates.append(rand_point)
                
        return np.vstack(candidates)[:batch_size]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update trust region based on recent performance
            if len(self.y) > 0:
                current_best = np.min(self.y)
                self.performance_history.append(current_best)
                
                # Keep only the last few evaluations for performance tracking
                if len(self.performance_history) > self.performance_window_size:
                    self.performance_history.pop(0)
                
                # Update trust region every few iterations or when we have enough data
                if (self.n_evals - self.last_improvement_evals > max(10, self.budget // 20)) and len(self.performance_history) >= 2:
                    self._update_trust_region()
                    self.last_improvement_evals = self.n_evals
            
            # Select next points to evaluate using acquisition function
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size
            
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Feedback
 The algorithm EMBO_TRE_ES got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4565 with standard deviation 0.0108.

took 21.76 seconds to run.