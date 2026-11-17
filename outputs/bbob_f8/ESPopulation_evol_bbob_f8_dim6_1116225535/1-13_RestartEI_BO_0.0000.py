from collections.abc import Callable
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
import numpy as np
import warnings


class RestartEI_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 2  # Initial sample size
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.best_x = None
        self.best_y = np.inf
        self.trust_radius = 0.5  # Initial trust region radius
        self.max_no_improvement = 5  # Threshold for restarting
        self.no_improvement_counter = 0
        self.restart_next = False  # Flag for global sampling in next iteration

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        # Anisotropic Matérn kernel with ARD
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds="fixed"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=3,
                random_state=int(self.rng.integers(2**31 - 1)),
            )
            gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu - self.best_y
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _select_next_points(self, model, batch_size=1):
        if self.restart_next:
            # Sample globally using Sobol
            candidates = self._sample_points(10 * self.dim)
            self.restart_next = False
        else:
            # Sample in trust region
            candidates = self.rng.uniform(
                low=self.best_x - self.trust_radius,
                high=self.best_x + self.trust_radius,
                size=(10 * self.dim, self.dim),
            )
        # Clip to bounds
        candidates = np.clip(candidates, self.bounds[0], self.bounds[1])

        # Evaluate EI and optimize
        ei = self._acquisition_function(candidates, model)
        best_idx = np.argmax(ei)
        x0 = candidates[best_idx]

        # Refine with L-BFGS-B
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1), model).ravel()[0]

        bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
        res = qmc.optimize(
            objective,
            bounds=bounds,
            method="L-BFGS-B",
            x0=x0,
            options={"maxiter": 50},
        )
        return res.x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        for i in range(len(X)):
            if y[i] < self.best_y:
                self.best_y = y[i]
                self.best_x = X[i].copy()
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.n_evals += self.n_init

        # Build initial model
        model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Select next point
            next_X = self._select_next_points(model)
            y_new = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, y_new)
            self.n_evals += len(next_X)

            # Update model
            model = self._fit_model(self.X, self.y)

            # Check for improvement
            improvement = self.best_y - y_new[0]
            if improvement > 1e-4:
                self.no_improvement_counter = 0
                # Adjust trust region
                self.trust_radius = min(1.2 * self.trust_radius, 2.0)
            else:
                self.no_improvement_counter += 1
                self.trust_radius = max(0.5 * self.trust_radius, 0.1)

            # Trigger restart if progress stalls
            if self.no_improvement_counter >= self.max_no_improvement:
                self.trust_radius = 0.5  # Reset trust radius
                self.restart_next = True
                self.no_improvement_counter = 0

        return self.best_y, self.best_x
