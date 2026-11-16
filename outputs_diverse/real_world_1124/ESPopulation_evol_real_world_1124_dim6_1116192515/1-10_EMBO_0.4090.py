from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import differential_evolution
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
        
        # Random seed generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.n_init = max(2 * dim + 1, 10)  # Initial points: more than d+1 but reasonable for small budgets
        
        # Adaptive exploration parameters
        self.exploration_weight_schedule = self._create_exploration_schedule()
        self.current_exploration_weight = 1.0
        
        # Acquisition function parameters
        self.acq_func_weights = [0.4, 0.3, 0.3]  # Weights for EI, LCB, Thompson sampling
        self.n_acq_restarts = min(20, max(5, dim))  # Number of acquisition function restarts
        
        # Model parameters
        self.model_n_restarts = 2  # Number of GP hyperparameter restarts
        self.model_max_train_size = 50  # Maximum training points for GP

    def _create_exploration_schedule(self):
        """Create a dynamic exploration weight schedule that decreases over time"""
        # Create schedule that decreases from 1.0 to 0.1 over the budget
        # More aggressive early exploration, then focused exploitation
        t = np.linspace(0, 1, self.budget)
        # Use a sigmoid-like decay function
        schedule = 1.0 - 0.9 * (1.0 / (1.0 + np.exp(5 * (t - 0.5))))
        return schedule

    def _sample_points(self, n_points):
        """Use Latin hypercube sampling with scrambling for good coverage"""
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng, scramble=True)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        """Fit a sparse Gaussian Process with ARD Matern kernel"""
        # Select subset of data for training to keep computational cost low
        if len(X) > self.model_max_train_size:
            train_indices = self.rng.choice(len(X), size=self.model_max_train_size, replace=False)
            X_train = X[train_indices]
            y_train = y[train_indices]
        else:
            X_train = X
            y_train = y
            
        # Use ARD Matern kernel with automatic relevance detection
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # Fit GP with noise estimation and optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=self.model_n_restarts,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        gp.fit(X_train, y_train)
        return gp

    def _acquisition_function(self, X):
        """Compute acquisition function values for candidates"""
        if self.X is None or self.y is None:
            # Return random acquisition values for initial points
            return np.random.rand(len(X), 1)
            
        # Fit model
        gp = self._fit_model(self.X, self.y)
        
        # Get predictions from GP
        mu, sigma = gp.predict(X, return_std=True)
        
        # Expected Improvement (EI)
        best_y = np.min(self.y)
        if sigma > 0:
            improvement = best_y - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        else:
            ei = np.zeros_like(mu)
        
        # Lower Confidence Bound (LCB)
        lcb = mu - self.current_exploration_weight * sigma
        
        # Thompson sampling: sample from posterior distribution
        samples = gp.sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
        thompson = -samples.flatten()  # Negative because we minimize
        
        # Combine acquisition functions with weights
        acq_values = (
            self.acq_func_weights[0] * ei.reshape(-1, 1) +
            self.acq_func_weights[1] * lcb.reshape(-1, 1) +
            self.acq_func_weights[2] * thompson.reshape(-1, 1)
        )
        
        return acq_values

    def _optimize_acquisition(self, candidates):
        """Optimize acquisition function with multiple restarts"""
        # Use differential evolution for global optimization of acquisition function
        n_candidates = len(candidates)
        
        # Select a subset of candidates for acquisition optimization to save computation
        if n_candidates > 100:
            selected_indices = self.rng.choice(n_candidates, size=100, replace=False)
            candidates_subset = candidates[selected_indices]
        else:
            candidates_subset = candidates
            
        # Use multiple restarts with different starting points
        best_acq_value = -np.inf
        best_point = None
        
        # Sample starting points from the candidates for multiple restarts
        restart_points = self._sample_points(self.n_acq_restarts)
        
        for i in range(self.n_acq_restarts):
            # Use one of the candidate points as starting point
            start_point = restart_points[i % len(restart_points)]
            
            # Simple local optimization using scipy's differential evolution
            def obj(x):
                x = np.array(x).reshape(1, -1)
                acq_val = self._acquisition_function(x)
                return -float(acq_val)  # Negative because we minimize
            
            # Optimize with bounds
            try:
                result = differential_evolution(
                    obj, 
                    [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)],
                    maxiter=20,
                    seed=int(self.rng.integers(2**31 - 1)),
                    popsize=5
                )
                
                if result.success:
                    acq_val = -result.fun
                    if acq_val > best_acq_value:
                        best_acq_value = acq_val
                        best_point = result.x.reshape(1, -1)
            except:
                # If optimization fails, continue with next restart
                continue
        
        return best_point if best_point is not None else candidates[0:1]

    def _select_next_points(self, batch_size):
        """Select next points to evaluate"""
        # Sample candidates for acquisition function optimization
        n_candidates = min(500, max(50, self.budget - self.n_evals))
        
        # Generate initial candidates using Latin hypercube sampling
        candidates = self._sample_points(n_candidates)
        
        # Optimize acquisition function to find best point(s)
        best_point = self._optimize_acquisition(candidates)
        
        # For batch selection, generate diverse points around the best found point
        if batch_size > 1:
            # Generate additional points in a neighborhood of the best point
            points = [best_point[0]]
            
            # Add some diversity by sampling nearby points
            for _ in range(batch_size - 1):
                # Perturb the best point slightly
                perturbation = self.rng.normal(0, 0.1, self.dim)
                neighbor = np.clip(best_point[0] + perturbation, self.bounds[0], self.bounds[1])
                points.append(neighbor)
            
            selected_points = np.array(points)
        else:
            selected_points = best_point
            
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        if len(X) == 0:
            return np.array([])
            
        y = []
        for point in X:
            try:
                val = func(point)
                y.append(val)
            except Exception as e:
                # Handle function evaluation errors by returning large value
                y.append(1e10)
                
        y = np.array(y).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            
        # Update exploration weight based on current iteration
        if self.n_evals < self.budget:
            self.current_exploration_weight = self.exploration_weight_schedule[self.n_evals]
        else:
            self.current_exploration_weight = 0.1  # Minimum exploration weight

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        # Initialize with Latin hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # If we're running low on budget, evaluate more points in parallel
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Evaluate these points
            next_values = self._evaluate_points(func, next_points)
            
            # Update the evaluation points
            self._update_eval_points(next_points, next_values)
            
            # Check if we've exhausted the budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
