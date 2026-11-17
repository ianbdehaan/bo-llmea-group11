from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings

class TrustRegionSobolMaternEIBOPlus:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 3 * dim)  # Larger initial design for better model warmup
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.eval_history = []

    def _sample_points(self, n_points, low=None, high=None):
        """Generate diverse candidates within bounds."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        # Mix Sobol and random restarts for diversity
        sobol_samples = qmc.scale(self.sobol_gen.random(n_points), low, high)
        random_samples = self.rng.uniform(low, high, (n_points, self.dim))
        return np.vstack([sobol_samples, random_samples])

    def _fit_model(self, X, y):
        """Fit GP with ARD Matérn kernel and limited hyperparameter restarts."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,  # Reduced from 3 to save time
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization with EI computation."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0
        return ei

    def _select_next_points(self):
        """Optimize acquisition function with L-BFGS-B and trust-region bounds."""
        if self.tr_center is None:
            return self._sample_points(10).reshape(-1, self.dim)  # Fallback to global search
        
        def constrained_acquisition(x):
            return -self._acquisition_function(x.reshape(1, -1))[0]  # Minimize negative EI
        
        # Generate trust-region bounds
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        bounds = [(l, h) for l, h in zip(low, high)]
        
        # Generate 100 random restarts within trust region
        x0 = self.rng.uniform(low, high, (100, self.dim))
        results = []
        for x in x0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    constrained_acquisition,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-6, 'maxiter': 50}
                )
                if res.success:
                    results.append((res.x, -res.fun))
        
        if not results:
            return self._sample_points(3).reshape(-1, self.dim)
        
        # Select top 3 candidates with highest EI values
        candidates = np.array([x for x, _ in results])
        acq_vals = np.array([val for _, val in results])
        best_indices = np.argsort(-acq_vals)[:3]
        return candidates[best_indices]

    def _update_tr_radius(self, improved):
        """Adjust trust region radius based on historical improvement."""
        self.eval_history.append(improved)
        if len(self.eval_history) < 3:
            return  # Wait for history to build
        recent_improvements = self.eval_history[-3:]
        improvement_count = sum(recent_improvements)
        
        if improvement_count >= 2:
            # Expand trust region when consistent improvement
            self.tr_radius = min(self.tr_radius * 1.2, self.max_tr_radius)
        elif improvement_count == 0:
            # Contract trust region when no improvement
            self.tr_radius = max(self.tr_radius * 0.8, self.min_tr_radius)
        # Else: maintain current radius

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial design
        X_init = self._sample_points(self.n_init)
        y_init = np.array([func(x) for x in X_init])
        self._update_eval_points(X_init, y_init)
        
        # Initialize trust region
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        self.tr_radius = 2.5  # Initial radius
        self.max_tr_radius = 5.0
        self.min_tr_radius = 0.1
        
        # Initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main loop
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = np.array([func(x) for x in next_X])
            self._update_eval_points(next_X, next_y)
            
            # Update trust region based on history
            improved = next_y[0] < self.y[best_idx]
            self._update_tr_radius(improved)
            
            # Update best solution
            best_idx = np.argmin(self.y)
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model every 3 evaluations to reduce computation
            if (self.n_evals % 3) == 0:
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
    
    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history with new points."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            self.n_evals += len(new_X)
