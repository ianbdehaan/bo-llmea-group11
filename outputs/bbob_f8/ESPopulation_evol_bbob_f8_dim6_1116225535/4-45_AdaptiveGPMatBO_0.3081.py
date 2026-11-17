from collections.abc import Callable
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import numpy as np

class AdaptiveGPMatBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Maximin design for diversity
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.best_y = float('inf')
        self.best_x = None
        self.model = None
        self.iter_count = 0

    def _sample_maximin_points(self, n_points):
        """Generate Latin Hypercube samples with maximin distance."""
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        """GP with Matérn 5/2 kernel and ARD."""
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6 * np.var(y) + 1e-12,  # Numerical stability
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0,  # Avoid time-consuming hyperparameter tuning
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Hybrid acquisition: EI for early stages, LCB for later stages."""
        stage = self.iter_count / self.budget  # Progress ratio
        if stage < 0.6:  # 60% of budget: EI
            mu, sigma = self.model.predict(X, return_std=True)
            f_best = self.best_y
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = f_best - mu
                Z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[imp < 0] = 0.0
            return ei
        else:  # 40% of budget: LCB with decreasing exploration
            mu, sigma = self.model.predict(X, return_std=True)
            exploration_weight = 0.8 * (1 - stage)  # Decay to 0 at end
            lcb = mu - exploration_weight * sigma
            return -lcb  # Minimize LCB

    def _select_next_point(self):
        """Generate global and local candidates, optimize acquisition."""
        n_candidates = 30  # Global candidates
        global_candidates = self._sample_maximin_points(n_candidates)
        local_candidates = self._sample_around_best(10)  # Local candidates near best
        all_candidates = np.vstack([global_candidates, local_candidates])

        # Evaluate acquisition function
        acq_values = self._acquisition_function(all_candidates)
        top_indices = np.argsort(acq_values, axis=0)[:n_candidates]  # Top 30
        top_candidates = all_candidates[top_indices]

        # Local refinement
        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return all_candidates[[0]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmin(acq_values_opt)  # Minimize LCB, maximize EI
        return optimized_points[best_idx].reshape(1, -1)

    def _sample_around_best(self, n_points):
        """Generate candidates around the current best solution."""
        if self.best_x is None:
            return self._sample_maximin_points(n_points)
        delta = 0.2 * (self.bounds[1] - self.bounds[0])  # 20% of domain
        x_center = self.best_x
        samples = self.rng.normal(loc=x_center, scale=delta, size=(n_points, self.dim))
        samples = np.clip(samples, self.bounds[0], self.bounds[1])
        return samples

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        y = np.zeros((n_new, 1))
        for i in range(n_new):
            y[i, 0] = func(X[i])
        self.n_evals += n_new
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
        # Update best
        min_idx = np.argmin(self.y)
        self.best_y = self.y[min_idx][0]
        self.best_x = self.X[min_idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        self.X = self._sample_maximin_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self._update_eval_points(self.X, self.y)

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            self.iter_count += 1
            next_point = self._select_next_point()
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Re-fit model every 5 iterations for efficiency
            if self.iter_count % 5 == 0:
                self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
