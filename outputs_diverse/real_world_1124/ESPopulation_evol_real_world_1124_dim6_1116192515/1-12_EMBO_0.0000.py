from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings

class EMBO:
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
        
        # Surrogate models
        self.gp = None
        self.rf = None
        
        # Acquisition function parameters
        self.acq_functions = ['ei', 'lcb', 'thompson']
        self.current_acq_idx = 0
        self.acq_weight_schedule = np.linspace(2.0, 0.2, max(10, budget // 10))  # Dynamic exploration weight
        
        # Trust region parameters
        self.trust_region_radius = 2.0
        self.trust_region_center = None
        
        # Model selection parameters
        self.model_performance = {'gp': 0.0, 'rf': 0.0}
        self.model_selection_threshold = 0.1
        self.model_history = []
        
        # Batch parameters
        self.batch_size = min(5, max(1, budget // 20))
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_models(self, X, y):
        # Fit both Gaussian Process and Random Forest models
        # GP model with ARD Matern kernel
        gp_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Use few restarts to keep computational cost low
        gp = GaussianProcessRegressor(
            kernel=gp_kernel,
            n_restarts_optimizer=min(3, max(1, self.n_evals // 10)),
            alpha=1e-6,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X, y)
        
        # RF model
        rf = RandomForestRegressor(
            n_estimators=min(50, max(10, self.n_evals // 2)),
            max_depth=min(10, max(3, self.n_evals // 20)),
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        
        rf.fit(X, y)
        
        return gp, rf

    def _acquisition_function(self, X):
        # Use current acquisition function based on model performance and problem characteristics
        if self.gp is None or self.rf is None:
            # For initial points before first model fit, use a simple heuristic
            return np.ones((X.shape[0], 1)) * -1e6
            
        # Get predictions from both models
        gp_mean, gp_std = self.gp.predict(X, return_std=True)
        rf_pred = self.rf.predict(X)
        
        # Current acquisition function based on portfolio switching
        current_acq = self.acq_functions[self.current_acq_idx]
        
        if current_acq == 'ei':
            # Expected Improvement
            best_y = np.min(self.y)
            if gp_std > 1e-8:
                improvement = best_y - gp_mean
                z = improvement / gp_std
                ei = improvement * norm.cdf(z) + gp_std * norm.pdf(z)
                return ei.reshape(-1, 1)
            else:
                return np.zeros((X.shape[0], 1))
                
        elif current_acq == 'lcb':
            # Lower Confidence Bound with adaptive exploration weight
            if len(self.acq_weight_schedule) > 0:
                # Use the last available weight or interpolate based on evaluation count
                current_weight_idx = min(len(self.acq_weight_schedule) - 1, self.n_evals // max(1, self.budget // len(self.acq_weight_schedule)))
                exploration_weight = self.acq_weight_schedule[current_weight_idx]
            else:
                exploration_weight = 0.5
                
            lcb = gp_mean - exploration_weight * gp_std
            return lcb.reshape(-1, 1)
            
        elif current_acq == 'thompson':
            # Thompson Sampling
            samples = self.gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
            return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Multi-start optimization of acquisition function with diverse starting points
        n_starts = min(30, max(10, batch_size * 5))  # More starts than needed to ensure diversity
        
        # Generate diverse starting points
        start_points = self._sample_points(n_starts)
        
        # Add trust region center if available and not too far from bounds
        if self.trust_region_center is not None:
            dist_to_center = cdist(start_points, self.trust_region_center.reshape(1, -1))
            # Only add points that are within a reasonable distance to the trust region center
            mask = dist_to_center.flatten() < 2.0 * self.trust_region_radius
            if np.sum(mask) > 0:
                start_points = np.vstack([start_points, self.trust_region_center])
        
        best_value = np.inf
        best_point = None
        
        # Evaluate acquisition function at all starting points and select best
        for i in range(n_starts):
            start_point = start_points[i:i+1]
            
            # Use a simple local search around this point to improve it
            try:
                result = differential_evolution(
                    lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0],
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    maxiter=20,
                    popsize=5,
                    seed=int(self.rng.integers(2**31 - 1)),
                    disp=False
                )
                
                if result.success:
                    point = result.x.reshape(1, -1)
                    acq_value = self._acquisition_function(point)[0, 0]
                    
                    if acq_value < best_value:
                        best_value = acq_value
                        best_point = point
                        
            except Exception:
                # If optimization fails, use the starting point directly
                continue
                
        # If we found a good point, return it; otherwise, return random points
        if best_point is not None:
            selected_points = best_point
        else:
            # Fallback: select from initial set
            selected_points = self._sample_points(1)
            
        # Return multiple points by sampling diverse directions
        candidates = [selected_points]
        
        # Add some diversity with additional random points
        if batch_size > 1:
            for _ in range(min(batch_size - 1, 5)):
                rand_point = self._sample_points(1)
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

    def _update_trust_region(self):
        # Update trust region based on recent progress
        if self.X is not None and len(self.X) >= 2:
            # Find the best point so far
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Set trust region center to best point
            self.trust_region_center = best_x
            
            # Update radius based on recent progress
            if len(self.X) > 10:
                # Calculate average distance between recent points
                recent_points = self.X[max(0, len(self.X)-5):]
                if len(recent_points) > 1:
                    distances = cdist(recent_points, recent_points)
                    np.fill_diagonal(distances, np.inf)
                    avg_dist = np.min(distances, axis=1).mean()
                    self.trust_region_radius = max(0.5, min(2.0, avg_dist))
            else:
                # Start with a large radius
                self.trust_region_radius = 2.0

    def _switch_acquisition_function(self):
        # Switch acquisition function based on performance
        if len(self.model_history) < 3:
            return
            
        # Simple switching logic: if one model consistently outperforms others, use it more often
        recent_performance = self.model_history[-3:]
        gp_performance = sum([1 for x in recent_performance if x == 'gp']) / len(recent_performance)
        rf_performance = sum([1 for x in recent_performance if x == 'rf']) / len(recent_performance)
        
        # Switch to the better performing model with some probability
        if gp_performance > rf_performance + 0.2:
            self.current_acq_idx = 0  # Use EI
        elif rf_performance > gp_performance + 0.2:
            self.current_acq_idx = 1  # Use LCB
        else:
            # Randomly switch between models with some probability
            if self.rng.random() < 0.3:  # 30% chance to change
                self.current_acq_idx = self.rng.integers(3)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube design
        init_points = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_y)
        
        # Fit initial models
        self.gp, self.rf = self._fit_models(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Update trust region
            self._update_trust_region()
            
            # Switch acquisition function occasionally
            if self.n_evals % 20 == 0 and self.n_evals > 0:
                self._switch_acquisition_function()
            
            # Select next points to evaluate using acquisition function
            batch_size = min(self.batch_size, self.budget - self.n_evals)  # Limit batch size
            
            # Update models with current data
            self.gp, self.rf = self._fit_models(self.X, self.y)
            
            # Select next points
            next_points = self._select_next_points(batch_size)
            
            # Evaluate selected points
            next_y = self._evaluate_points(func, next_points)
            
            # Update evaluation history
            self._update_eval_points(next_points, next_y)
            
            # Record model performance for switching
            if len(self.X) > 10:
                best_current = np.min(self.y)
                if self.model_history:
                    prev_best = min(self.model_history[-5:])
                    if best_current < prev_best - 0.01:  # Significant improvement
                        # Add to performance history (randomly select model for now)
                        self.model_history.append('gp' if self.rng.random() > 0.5 else 'rf')
                    else:
                        # No significant improvement, continue with current model
                        pass
                else:
                    self.model_history.append('gp')  # Default initial selection
            
            # If we're close to budget, stop early to avoid unnecessary evaluations
            if self.n_evals >= self.budget - 1:
                break
        
        # Return best result found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
