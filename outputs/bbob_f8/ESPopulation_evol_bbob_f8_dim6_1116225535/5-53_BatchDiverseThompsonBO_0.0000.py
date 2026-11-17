from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BatchDiverseThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.batch_size = 5  # Match 10-core parallelism

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            optimizer='fmin_l_bfgs_b',
            alpha=1e-6,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_batch(self):
        candidates = self._sample_points(50)  # Global sampling
        acq_values = self._acquisition_function(candidates).flatten()
        
        # Greedy maximin selection of 5 diverse points
        idxs = np.argsort(acq_values)[-10:]  # Top 10 candidates
        selected_idx = [idxs[0]]
        remaining_idx = idxs[1:].tolist()
        
        for _ in range(self.batch_size - 1):
            if not remaining_idx:
                break  # Fill with random if fewer than 5 candidates
            distances = np.min(np.linalg.norm(candidates[remaining_idx] - candidates[selected_idx], axis=1), axis=0)
            next_idx = remaining_idx[np.argmax(distances)]
            selected_idx.append(next_idx)
            remaining_idx.remove(next_idx)
        
        return candidates[selected_idx]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            batch = self._select_batch()
            new_y = self._evaluate_points(func, batch)
            self._update_eval_points(batch, new_y)
            
            if np.any(new_y < best_y):
                best_y = np.min(new_y)
                best_x = batch[np.argmin(new_y)]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
