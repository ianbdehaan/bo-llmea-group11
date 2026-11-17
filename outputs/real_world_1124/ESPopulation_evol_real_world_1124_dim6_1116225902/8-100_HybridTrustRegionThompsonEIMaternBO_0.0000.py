from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import minimize, Bounds
import math

class HybridTrustRegionThompsonEIMaternBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Adaptive initial design
        self.refit_interval = 1  # Dynamic based on progress
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        self.model = None
        self.best_X = None
        self.region_radius = 1.0  # Initial trust region radius
        self.batch_size = 2  # Adaptive to evaluation progress
        self.improvement_history = []  # Track recent improvements

    def _sample_points(self, n_points, trust_center=None, radius=None):
        if trust_center is None:
            # Global sampling using Sobol + Latin Hypercube hybrid
            sobol = self.sobol_gen.random(n_points // 2)
            lhs = qmc.LatinHypercube(d=self.dim, seed=int(self.rng.integers(2**31 - 1))).random(n_points // 2)
            samples = np.vstack([sobol, lhs])
        else:
            # Local sampling within trust region using uniform hypercube
            lower = np.clip(trust_center - radius, -5.0, 5.0)
            upper = np.clip(trust_center + radius, -5.0, 5.0)
            samples = self.rng.uniform(low=lower, high=upper, size=(n_points, self.dim))
        return samples

    def _fit_model(self, X, y):
        # Log-transform for stability with non-negative objectives
        y_log = np.log(y + 1e-8)
        
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y_log)
        return model

    def _select_next_points(self, use_thompson=True):
        # Generate adaptive number of candidates based on dimension
        n_candidates = max(50, int(200 * math.sqrt(2 * self.dim)))
        candidates = self._sample_points(n_candidates, self.best_X, self.region_radius)
        
        if use_thompson:
            # Thompson Sampling with ensemble of 20 function samples
            sampled_funcs = self.model.sample_y(candidates, n_samples=20)
            idxs = np.argmin(sampled_funcs, axis=1)
            
            # Select top 25% of candidates
            top_indices = np.argsort(sampled_funcs.mean(axis=1))[:n_candidates//4]
            top_candidates = candidates[top_indices]
            
            # Cluster with MiniBatchKMeans for diverse batch
            if len(top_candidates) >= self.batch_size:
                kmeans = MiniBatchKMeans(n_clusters=self.batch_size, n_init=1, batch_size=100, random_state=int(self.rng.integers(2**31 - 1)))
                kmeans.fit(top_candidates)
                next_X = kmeans.cluster_centers_
            else:
                next_X = top_candidates[:self.batch_size]
        else:
            # Expected Improvement acquisition
            y_mean, y_std = self.model.predict(candidates, return_std=True)
            best_y = np.min(self.y)
            
            # Compute EI
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = y_mean - best_y
                z = imp / y_std
                ei = (imp * norm.cdf(z)) + (y_std * norm.pdf(z))
                ei[np.isnan(ei)] = 0.0
            
            # Optimize EI with L-BFGS-B from diverse starting points
            init_points = self._sample_points(50, self.best_X, self.region_radius)
            res = []
            for x0 in init_points:
                bounds = Bounds(-5.0, 5.0)
                r = minimize(lambda x: -ei[np.argmin(np.linalg.norm(candidates - x, axis=1))], 
                            x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                if r.success:
                    res.append((r.x, -r.fun))
            res = np.array(res)
            next_X = res[res[:,1].argsort()[:self.batch_size], 0]
        
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
            self.best_X = new_X[np.argmin(new_y)].copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
            if np.min(new_y) < np.min(self.y):
                self.best_X = new_X[np.argmin(new_y)].copy()
                self.improvement_history.append(True)
                # Expand trust region gradually
                alpha = 0.9 if self.region_radius < 2.0 else 0.95
                self.region_radius = alpha * self.region_radius * 1.1
            else:
                self.improvement_history.append(False)
                # Contract trust region exponentially
                alpha = 0.8 if self.region_radius > 0.3 else 0.9
                self.region_radius = alpha * self.region_radius * 0.9

        # Clean up old history to prevent memory issues
        if len(self.improvement_history) > 5:
            self.improvement_history.pop(0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        # Main loop with alternating acquisition functions
        iteration = 0
        use_thompson = True  # Alternate between Thompson and EI
        while self.n_evals < self.budget:
            # Alternate acquisition function every 2 iterations
            use_thompson = not use_thompson if iteration % 2 == 0 else use_thompson
            
            # Refit model when improvement is detected or on fixed interval
            if (self.improvement_history[-1] if len(self.improvement_history) > 0 else True) or iteration % 2 == 0:
                self.model = self._fit_model(self.X, self.y)
            
            next_X = self._select_next_points(use_thompson)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
