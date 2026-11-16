from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class AdaptiveMultiSurrogateBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(2 * dim + 1, 10)  # Initial points - start with more points for higher dimensions

        # Random number generator
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Surrogate model management
        self.current_model = "gp"  # Start with GP
        self.model_performance = {"gp": [], "rf": []}
        self.model_switch_threshold = 0.5  # Threshold for switching models
        
        # Acquisition function parameters
        self.acq_func_type = "hybrid"  # Can be "ei", "hybrid"
        
        # Exploration strategy
        self.local_exploration_factor = 0.1  # Factor for local exploration term
        self.progressive_refinement = True
        self.refinement_threshold = 0.05  # Threshold for refinement
        
        # History tracking
        self.best_y_history = []
        self.iteration = 0

    def _sample_points(self, n_points):
        # Sample points using maximin Latin hypercube sampling for good space-filling properties
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + sample * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_gp_model(self, X, y):
        # Fit Gaussian Process surrogate model with ARD Matern kernel
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,  # Limited restarts for computational efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _fit_rf_model(self, X, y):
        # Fit Random Forest surrogate model
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # For reproducibility
        )
        rf.fit(X, y)
        return rf

    def _predict_gp(self, model, X):
        # Predict with GP model and return mean and std
        mu, sigma = model.predict(X, return_std=True)
        return mu, sigma

    def _predict_rf(self, model, X):
        # Predict with RF model (return mean only for simplicity)
        # RF doesn't naturally provide uncertainty estimates, so we'll estimate from the forest variance
        predictions = np.array([tree.predict(X) for tree in model.estimators_])
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        return mu, sigma

    def _acquisition_function(self, X, gp_model, rf_model, best_y):
        # Hybrid acquisition function combining Expected Improvement with local exploration
        if self.current_model == "gp":
            mu, sigma = self._predict_gp(gp_model, X)
        else:
            mu, sigma = self._predict_rf(rf_model, X)
            
        # Expected improvement
        improvement = mu - best_y
        improvement = np.maximum(improvement, 0)  # Ensure non-negative
        
        # Local exploration term (based on uncertainty)
        local_explore = self.local_exploration_factor * sigma
        
        # Combined acquisition function
        if self.acq_func_type == "hybrid":
            # Hybrid: EI + local exploration term
            acq_value = improvement + local_explore
        else:
            # Simple expected improvement
            acq_value = improvement
            
        return acq_value.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using multi-start L-BFGS with restarts
        
        # Sample random points for diversity
        n_random = max(5, batch_size // 2)
        random_points = self._sample_points(n_random)
        
        # Optimize acquisition function using L-BFGS-B from multiple starting points
        if self.X is not None and len(self.X) > 0:
            best_y = np.min(self.y)
            
            # Fit both models
            gp_model = self._fit_gp_model(self.X, self.y)
            rf_model = self._fit_rf_model(self.X, self.y)
            
            # Generate starting points for optimization
            n_starts = min(10, max(3, batch_size // 2))
            start_points = self._sample_points(n_starts)
            
            best_acq_val = -np.inf
            best_opt_point = None
            
            for start_point in start_points:
                def obj(x):
                    x = x.reshape(1, -1)
                    acq_val = self._acquisition_function(x, gp_model, rf_model, best_y)[0, 0]
                    return -acq_val  # Negative because we minimize
                
                try:
                    res = minimize(obj, start_point, method='L-BFGS-B', 
                                 bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                                 options={'maxiter': 50})
                    if -res.fun > best_acq_val:
                        best_acq_val = -res.fun
                        best_opt_point = res.x
                except:
                    continue
            
            if best_opt_point is not None:
                optimize_points = best_opt_point.reshape(1, -1)
            else:
                optimize_points = self._sample_points(1)
        else:
            optimize_points = self._sample_points(1)
        
        # Combine points
        if len(optimize_points) > 0:
            selected_points = np.vstack([random_points, optimize_points])
        else:
            selected_points = random_points
        
        # Limit to batch_size
        if len(selected_points) > batch_size:
            selected_points = selected_points[:batch_size]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y_values = []
        for point in X:
            y_values.append(func(point))
        
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _switch_model(self):
        # Switch model based on performance history
        if len(self.model_performance["gp"]) >= 3 and len(self.model_performance["rf"]) >= 3:
            gp_perf = np.mean(self.model_performance["gp"][-3:])
            rf_perf = np.mean(self.model_performance["rf"][-3:])
            
            # Switch if RF performs significantly better (based on recent performance)
            if rf_perf < gp_perf * self.model_switch_threshold:
                self.current_model = "rf"
            elif gp_perf < rf_perf * self.model_switch_threshold:
                self.current_model = "gp"

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            batch_size = min(5, self.budget - self.n_evals)  # Limit batch size to remaining budget
            if batch_size <= 0:
                break
                
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate new points
            next_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_y)
            
            # Update model performance metrics (simplified - in practice would be more sophisticated)
            if len(self.y) > 1:
                current_best = np.min(self.y)
                self.best_y_history.append(current_best)
                
                # Update model performance tracking
                if self.iteration >= 3:  # Only start tracking after some evaluations
                    prev_best = self.best_y_history[-2]
                    improvement = (prev_best - current_best) / max(prev_best, 1e-8)
                    if self.current_model == "gp":
                        self.model_performance["gp"].append(improvement)
                    else:
                        self.model_performance["rf"].append(improvement)
                    
                    # Check if we should switch models
                    if len(self.best_y_history) > 5 and self.iteration % 3 == 0:
                        self._switch_model()
            
            self.iteration += 1
            
            # Early termination for small budgets to avoid excessive overhead
            if self.n_evals >= self.budget - 1:
                break
                
        # Return the best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
