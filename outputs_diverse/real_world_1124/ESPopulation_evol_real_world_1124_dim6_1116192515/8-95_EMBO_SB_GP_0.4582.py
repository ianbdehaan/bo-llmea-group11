from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class EMBO_SB_GP:
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
        
        # Acquisition function parameters
        # Dynamic exploration weight based on progress through budget
        self.acq_weight_schedule = np.linspace(3.0, 0.1, max(10, budget // 10))  
        
        # Trust region parameters
        self.trust_region_radius = 2.0
        self.trust_region_shrink_factor = 0.8
        self.trust_region_expand_factor = 1.2
        self.min_trust_region = 0.1
        self.max_trust_region = 3.0
        
        # For trust region management
        self.best_so_far = None
        self.progress_counter = 0
        self.consecutive_no_improvement = 0
        self.max_consecutive_no_improvement = 5

    def _sample_points(self, n_points):
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        # Fit a Gaussian Process regressor with ARD Matern kernel
        # Use a small number of hyperparameter restarts for efficiency
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Few restarts for efficiency
            alpha=1e-6,  # Small noise term
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y.flatten())
        return gp

    def _acquisition_function(self, X):
        # Expected Improvement with adaptive exploration weight using GP predictions
        if self.gp is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from GP (mean and std)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Find the best observed value so far
        if self.y is not None and len(self.y) > 0:
            best_y = np.min(self.y)
        else:
            best_y = 0.0
            
        # Adaptive exploration weight based on budget progress
        progress_ratio = min(1.0, self.n_evals / self.budget)
        if len(self.acq_weight_schedule) > 0:
            acq_weight = self.acq_weight_schedule[min(len(self.acq_weight_schedule)-1, int(progress_ratio * len(self.acq_weight_schedule)))]
        else:
            acq_weight = 1.0
            
        # Expected Improvement acquisition function
        improvement = best_y - mean
        z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Apply exploration weight to control exploitation vs exploration
        ei = ei * acq_weight
        
        return ei.reshape(-1, 1)

    def _optimize_acquisition(self, n_starts=20):
        """Optimize acquisition function using differential evolution"""
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        best_value = np.inf
        best_point = None
        
        # Define bounds for optimization
        bounds = list(zip(self.bounds[0], self.bounds[1]))
        
        # For small number of points, use simpler approach
        if n_starts <= 5:
            # Use simple grid search for few candidates
            candidates = start_points
            best_idx = np.argmin(self._acquisition_function(candidates).flatten())
            return candidates[best_idx:best_idx+1]
        
        # Use differential evolution for robust optimization
        try:
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]
                
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=int(self.rng.integers(2**31 - 1)),
                maxiter=50,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            if result.success:
                point = result.x.reshape(1, -1)
                return point
                
        except Exception as e:
            # Fallback to best starting point if optimization fails
            try:
                acq_values = self._acquisition_function(start_points).flatten()
                best_idx = np.argmin(acq_values)
                return start_points[best_idx:best_idx+1]
            except Exception:
                # Fallback to random point
                return self._sample_points(1)
        
        # Fallback to best starting point if differential evolution fails
        try:
            acq_values = self._acquisition_function(start_points).flatten()
            best_idx = np.argmin(acq_values)
            return start_points[best_idx:best_idx+1]
        except Exception:
            # Final fallback to random point
            return self._sample_points(1)

    def _select_next_points(self, batch_size):
        """Select next points using acquisition function optimization"""
        if batch_size <= 2:
            selected_points = []
            for _ in range(batch_size):
                # Use differential evolution to optimize acquisition function
                best_point = self._optimize_acquisition(n_starts=10)
                if best_point is not None:
                    selected_points.append(best_point)
                else:
                    # Fallback to random point
                    selected_points.append(self._sample_points(1))
            return np.vstack(selected_points)
        else:
            # For larger batches, use a combination approach
            selected_points = []
            
            # First select one point using acquisition optimization
            best_point = self._optimize_acquisition(n_starts=20)
            if best_point is not None:
                selected_points.append(best_point)
            
            # Then add diverse points using trust region sampling
            remaining = batch_size - len(selected_points)
            if remaining > 0:
                # Sample from trust region around the current best point
                if self.X is not None and len(self.X) > 0:
                    center = self.X[np.argmin(self.y)]
                    # Sample within trust region with some randomness
                    for _ in range(remaining):
                        # Generate points within trust region
                        sample_point = center + self.rng.uniform(-self.trust_region_radius, self.trust_region_radius, self.dim)
                        # Clip to bounds
                        sample_point = np.clip(sample_point, self.bounds[0], self.bounds[1])
                        selected_points.append(sample_point.reshape(1, -1))
                else:
                    # Fallback to random sampling
                    for _ in range(remaining):
                        selected_points.append(self._sample_points(1))
            
            return np.vstack(selected_points)

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

    def _update_trust_region(self):
        """Update trust region based on progress"""
        if self.best_so_far is None:
            self.best_so_far = np.min(self.y) if self.y is not None and len(self.y) > 0 else float('inf')
            return
            
        current_best = np.min(self.y)
        if current_best < self.best_so_far:
            # Improvement made, potentially expand trust region
            self.consecutive_no_improvement = 0
            self.trust_region_radius = min(self.max_trust_region, 
                                          self.trust_region_radius * self.trust_region_expand_factor)
            self.best_so_far = current_best
        else:
            # No improvement, shrink trust region and potentially stop if needed
            self.consecutive_no_improvement += 1
            self.trust_region_radius = max(self.min_trust_region,
                                          self.trust_region_radius * self.trust_region_shrink_factor)
            
            if self.consecutive_no_improvement >= self.max_consecutive_no_improvement:
                # Stop early if no improvement for several iterations
                pass

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Sobol sequence design for better space-filling properties
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial model
        self.gp = self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Adaptive batch sizing based on remaining budget
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 5:
                batch_size = min(2, remaining_budget)
            elif remaining_budget <= 10:
                batch_size = min(3, remaining_budget)
            else:
                batch_size = min(5, remaining_budget // 4)  # Reduce batch size as we approach budget limit
            
            if batch_size <= 0:
                break
                
            # Update model with current data
            self.gp = self._fit_model(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # Update trust region based on progress
            self._update_trust_region()
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
