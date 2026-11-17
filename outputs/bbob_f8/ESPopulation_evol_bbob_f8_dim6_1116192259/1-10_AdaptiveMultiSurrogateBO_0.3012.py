from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution, minimize
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import norm

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
        self.n_init = max(2 * self.dim + 1, 10) # Initial points based on dimensionality
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Model and acquisition function parameters
        self.models = {
            'gp': None,
            'rf': None,
            'lr': None
        }
        self.model_performance = {'gp': 0, 'rf': 0, 'lr': 0}
        self.best_y = np.inf
        self.best_x = None
        self.current_model = 'gp'  # Start with GP
        self.scaler = StandardScaler()
        
        # Trust region parameters
        self.trust_region_radius = 1.0
        self.last_improvement_iter = 0
        self.iteration = 0
        
        # For restart detection
        self.stagnation_threshold = 5
        self.best_y_history = []
        
    def _sample_points(self, n_points):
        # Use maximin Latin Hypercube design for initial sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = qmc.scale(points, self.bounds[0], self.bounds[1])
        return points
    
    def _fit_model(self, model_type, X, y):
        """Fit a specific type of model"""
        if model_type == 'gp':
            try:
                kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                    random_state=int(self.rng.integers(2**31 - 1))
                )
                gp.fit(X, y)
                return gp
            except:
                return None
        elif model_type == 'rf':
            try:
                rf = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=int(self.rng.integers(2**31 - 1)),
                    n_jobs=1
                )
                rf.fit(X, y)
                return rf
            except:
                return None
        elif model_type == 'lr':
            try:
                lr = LinearRegression()
                lr.fit(X, y)
                return lr
            except:
                return None
        return None
    
    def _evaluate_model_performance(self, model_type, X_test, y_test):
        """Evaluate performance of a model on test data"""
        if model_type == 'gp':
            try:
                predictions = self.models[model_type].predict(X_test)
                mse = np.mean((predictions - y_test.flatten())**2)
                return -mse  # Negative MSE for maximization
            except:
                return -np.inf
        elif model_type == 'rf':
            try:
                predictions = self.models[model_type].predict(X_test)
                mse = np.mean((predictions - y_test.flatten())**2)
                return -mse
            except:
                return -np.inf
        elif model_type == 'lr':
            try:
                predictions = self.models[model_type].predict(X_test)
                mse = np.mean((predictions - y_test.flatten())**2)
                return -mse
            except:
                return -np.inf
        return -np.inf
    
    def _select_best_model(self, X_val, y_val):
        """Select the best performing model based on validation performance"""
        # Test all models
        best_performance = -np.inf
        best_model = 'gp'
        
        for model_type in ['gp', 'rf', 'lr']:
            if self.models[model_type] is not None:
                perf = self._evaluate_model_performance(model_type, X_val, y_val)
                if perf > best_performance:
                    best_performance = perf
                    best_model = model_type
        
        return best_model
    
    def _acquisition_function(self, X):
        """Ensemble acquisition function combining multiple strategies"""
        # Expected Improvement
        try:
            mean, std = self.models[self.current_model].predict(X, return_std=True)
            improvement = self.best_y - mean - 1e-8
            ei = improvement * norm.cdf(improvement/std) + std * norm.pdf(improvement/std)
        except:
            ei = np.ones((X.shape[0], 1)) * 0.1
            
        # Probability of Improvement
        try:
            mean, std = self.models[self.current_model].predict(X, return_std=True)
            pi = norm.cdf((self.best_y - mean) / (std + 1e-8))
        except:
            pi = np.ones((X.shape[0], 1)) * 0.5
            
        # Thompson Sampling (sample from GP posterior)
        try:
            if self.current_model == 'gp':
                samples = self.models[self.current_model].sample_y(X, n_samples=1, random_state=int(self.rng.integers(2**31 - 1)))
                ts = -samples.flatten().reshape(-1, 1)  # Negative because we minimize
            else:
                # For non-GP models, use variance as approximation
                predictions = np.array([self.models[self.current_model].predict(X) for _ in range(10)])
                variance = np.var(predictions, axis=0)
                ts = variance.reshape(-1, 1)
        except:
            ts = np.ones((X.shape[0], 1)) * 0.1
            
        # Combine acquisition functions with adaptive weights
        # Use higher weight for EI in early iterations, more exploration later
        if self.n_evals < self.budget * 0.3:
            weights = [0.6, 0.2, 0.2]  # More focus on EI in early phase
        else:
            weights = [0.3, 0.3, 0.4]  # More exploration later
            
        combined_acq = (weights[0] * ei + 
                       weights[1] * pi + 
                       weights[2] * ts)
        
        return combined_acq
    
    def _select_next_points(self, batch_size):
        """Select next points using hybrid acquisition optimization"""
        best_acq_val = -np.inf
        best_x = None
        
        # Try multiple starting points for differential evolution
        n_starts = min(10, max(3, self.n_evals // 5))
        
        for _ in range(n_starts):
            # Start from a random point within bounds
            start_point = self._sample_points(1)[0]
            
            def acq_func(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0][0]
            
            try:
                # Use differential evolution for global search
                res = differential_evolution(
                    acq_func,
                    bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                    seed=int(self.rng.integers(2**31 - 1)),
                    maxiter=20,
                    popsize=15
                )
                
                if res.success:
                    val = -res.fun
                    if val > best_acq_val:
                        best_acq_val = val
                        best_x = res.x.reshape(1, -1)
            except:
                continue
        
        # If no good points found with DE, try L-BFGS refinement from a few random points
        if best_x is None:
            for _ in range(5):
                start_point = self._sample_points(1)[0]
                
                def acq_func(x):
                    x = x.reshape(1, -1)
                    return -self._acquisition_function(x)[0][0]
                
                try:
                    res = minimize(
                        acq_func, 
                        start_point,
                        method='L-BFGS-B',
                        bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                        options={'maxiter': 20}
                    )
                    
                    if res.success:
                        val = -res.fun
                        if val > best_acq_val:
                            best_acq_val = val
                            best_x = res.x.reshape(1, -1)
                except:
                    continue
        
        # If still no good points, return a random point
        if best_x is None:
            return self._sample_points(batch_size)
        
        return np.repeat(best_x, batch_size, axis=0)
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None:
            return np.array([])
            
        y_vals = []
        for x in X:
            try:
                y_val = func(x)
                y_vals.append(y_val)
            except:
                # If evaluation fails, use a large value
                y_vals.append(np.inf)
        
        self.n_evals += len(X)
        return np.array(y_vals).reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
        
        # Update best point found so far
        min_idx = np.argmin(self.y)
        if self.y[min_idx] < self.best_y:
            self.best_y = self.y[min_idx]
            self.best_x = self.X[min_idx]
    
    def _update_trust_region(self):
        """Update trust region based on recent progress"""
        # Simple adaptive mechanism
        if len(self.best_y_history) >= 2:
            improvement = self.best_y_history[-2] - self.best_y_history[-1]
            if improvement < 1e-4:  # Very small improvement, shrink trust region
                self.trust_region_radius *= 0.9
            else:  # Significant improvement, expand trust region
                self.trust_region_radius = min(self.trust_region_radius * 1.1, 5.0)
    
    def _detect_stagnation(self):
        """Detect if optimization has stagnated"""
        if len(self.best_y_history) < self.stagnation_threshold:
            return False
            
        recent_improvements = [
            self.best_y_history[-i] - self.best_y_history[-i-1] 
            for i in range(1, min(self.stagnation_threshold, len(self.best_y_history)))
        ]
        
        avg_improvement = np.mean(recent_improvements)
        return avg_improvement < 1e-6
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with maximin Latin Hypercube design
        init_X = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_X)
        self._update_eval_points(init_X, init_y)
        
        # Fit initial models
        for model_type in ['gp', 'rf', 'lr']:
            self.models[model_type] = self._fit_model(model_type, self.X, self.y.flatten())
        
        # Select best model from initial data
        val_indices = self.rng.choice(len(self.X), size=max(5, len(self.X)//3), replace=False)
        X_val = self.X[val_indices]
        y_val = self.y[val_indices]
        self.current_model = self._select_best_model(X_val, y_val)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            self.iteration += 1
            
            # Update trust region and check for stagnation
            self.best_y_history.append(self.best_y)
            self._update_trust_region()
            
            if self._detect_stagnation():
                # Restart with a new set of points
                restart_points = self._sample_points(5)
                restart_y = self._evaluate_points(func, restart_points)
                self._update_eval_points(restart_points, restart_y)
                
                # Refit models after restart
                for model_type in ['gp', 'rf', 'lr']:
                    self.models[model_type] = self._fit_model(model_type, self.X, self.y.flatten())
                
                self.current_model = self._select_best_model(X_val, y_val)
                self.last_improvement_iter = self.iteration
                continue
            
            # If we're running low on budget, reduce batch size
            remaining_budget = self.budget - self.n_evals
            batch_size = min(5, remaining_budget)
            
            if batch_size <= 0:
                break
                
            # Select next points to evaluate
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            
            # Update the model with new data
            self._update_eval_points(next_X, next_y)
            
            # Re-fit models periodically (every 3 evaluations) to maintain accuracy
            if self.n_evals % 3 == 0:
                try:
                    for model_type in ['gp', 'rf', 'lr']:
                        self.models[model_type] = self._fit_model(model_type, self.X, self.y.flatten())
                    
                    # Select best model based on recent performance
                    self.current_model = self._select_best_model(X_val, y_val)
                except:
                    pass  # If fitting fails, keep current models
        
        return self.best_y, self.best_x
