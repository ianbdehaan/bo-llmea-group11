from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
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
        
        # Evolution strategy parameters
        self.es_population_size = min(20, max(5, dim * 2))
        self.es_num_generations = 5
        self.local_refinement_iterations = 3

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit and tune Gaussian Process surrogate model with adaptive noise"""
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
        """Expected Improvement acquisition function with adaptive exploration"""
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

    def _acquisition_optimizer_evolution_strategy(self, n_candidates=50):
        """Optimize acquisition function using small evolution strategy with local refinement"""
        # Generate initial population for evolution strategy
        if self.X is not None and len(self.X) > 0:
            # Start from current best point + random perturbations
            best_x = self.X[np.argmin(self.y)]
            population = []
            
            # Generate diverse starting points around the current best
            for _ in range(self.es_population_size):
                # Random perturbation around best point with trust region constraint
                if self.trust_region_radius > 0:
                    perturbation = (self.rng.random(self.dim) - 0.5) * 2 * self.trust_region_radius
                    candidate = np.clip(best_x + perturbation, self.bounds[0], self.bounds[1])
                else:
                    # If trust region is too small, use random sampling
                    candidate = self._sample_points(1)[0]
                
                population.append(candidate)
        else:
            # If no points yet, use Latin hypercube sampling
            population = self._sample_points(self.es_population_size)
        
        # Evolution strategy loop
        for generation in range(self.es_num_generations):
            # Evaluate acquisition function for all candidates
            acq_values = self._acquisition_function(np.array(population))
            
            # Sort by acquisition value (descending order)
            sorted_indices = np.argsort(acq_values.flatten())[::-1]
            top_candidates = [population[i] for i in sorted_indices[:self.es_population_size//2]]
            
            # Generate new population through mutation and crossover
            new_population = []
            
            # Keep best individuals
            new_population.extend(top_candidates)
            
            # Add mutated versions of the best individuals
            for _ in range(self.es_population_size - len(top_candidates)):
                if len(top_candidates) > 0:
                    parent = top_candidates[self.rng.integers(len(top_candidates))]
                    # Mutation with adaptive step size
                    mutation_strength = self.trust_region_radius / (generation + 1)
                    mutated = parent + self.rng.normal(0, mutation_strength, self.dim)
                    mutated = np.clip(mutated, self.bounds[0], self.bounds[1])
                    new_population.append(mutated)
                else:
                    # Fallback to random sampling
                    new_population.append(self._sample_points(1)[0])
            
            population = new_population
        
        # Local refinement using L-BFGS-B for the best candidate
        if len(population) > 0:
            best_candidate = population[0]
            acq_values = self._acquisition_function(np.array([best_candidate]))
            
            # Perform local refinement with L-BFGS-B
            for _ in range(self.local_refinement_iterations):
                try:
                    result = minimize(
                        lambda x: -self._acquisition_function(x.reshape(1, -1))[0][0],
                        best_candidate,
                        method='L-BFGS-B',
                        bounds=list(zip(self.bounds[0], self.bounds[1])),
                        options={'maxiter': 5}
                    )
                    if result.success:
                        refined_candidate = result.x
                        # Ensure it's within bounds
                        refined_candidate = np.clip(refined_candidate, self.bounds[0], self.bounds[1])
                        
                        # Check if refinement improved the acquisition value
                        current_acq = self._acquisition_function(best_candidate.reshape(1, -1))[0][0]
                        refined_acq = self._acquisition_function(refined_candidate.reshape(1, -1))[0][0]
                        
                        if refined_acq > current_acq:
                            best_candidate = refined_candidate
                except:
                    # If local refinement fails, keep original
                    pass
                
                # Update with new candidate
                acq_values = self._acquisition_function(np.array([best_candidate]))
                
            return np.array([best_candidate])
        
        return np.array([self._sample_points(1)[0]])

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using evolution strategy optimizer"""
        # Use evolution strategy with local refinement for acquisition function optimization
        candidates = self._acquisition_optimizer_evolution_strategy(n_candidates=50)
        
        return candidates

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
            
            # Select next points using acquisition function optimizer
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
