from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import MiniBatchKMeans

class EnsembleThompsonTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2*dim
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.gp_ensemble = [None]*8  # 8 GPs with diverse kernels
        self.best_point: np.ndarray = None
        self.trust_radius = 5.0  # Initial trust region radius

    def _sample_points(self, n_points):
        # Latin hypercube for better initial coverage
        lhs = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = lhs.random(n=n_points)
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _initialize_gp(self, X, y, idx):
        # Create diverse GPs with different kernel parameters
        kernel = 1.0 * Matern(nu=5/2, length_scale=np.clip(1.0 + 0.5*(idx-4), 0.2, 3.0),
                             length_scale_bounds=(0.1, 5.0))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6 + 5e-7*idx,
            n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        gp.fit(X, y)
        return gp

    def _fit_ensemble(self, X, y, iteration):
        # Refit every 3 iterations to balance cost and accuracy
        if iteration % 3 == 0:
            for i in range(8):
                self.gp_ensemble[i] = self._initialize_gp(X, y, i)

    def _select_next_points(self, batch_size):
        current_lower = np.maximum(self.bounds[0], self.best_point - self.trust_radius)
        current_upper = np.minimum(self.bounds[1], self.best_point + self.trust_radius)
        
        # Generate diverse candidates within trust region
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        candidates = sobol.random(n=200)
        candidates = current_lower + candidates * (current_upper - current_lower)
        
        # Thompson Sampling from ensemble
        thompson_candidates = []
        for gp in self.gp_ensemble:
            # Sample from GP posterior
            mean, std = gp.predict(candidates, return_std=True)
            thompson_samples = mean + std * self.rng.normal(size=mean.shape)
            thompson_candidates.append(thompson_samples)
        
        # Stack all Thompson samples and cluster to find diverse batch
        thompson_candidates = np.stack(thompson_candidates, axis=1)
        flat_candidates = thompson_candidates.reshape(-1, self.dim)
        if flat_candidates.shape[0] < batch_size:
            return flat_candidates
            
        # Max-min clustering to maintain diversity
        kmeans = MiniBatchKMeans(n_clusters=batch_size, batch_size=100, 
                                n_init=1, random_state=int(self.rng.integers(2**31 - 1)))
        kmeans.fit(flat_candidates)
        cluster_centers = kmeans.cluster_centers_
        
        # Select best cluster center (lowest mean prediction) for exploitation
        cluster_means = np.mean([gp.predict(cluster_centers) for gp in self.gp_ensemble], axis=0)
        best_idx = np.argmin(cluster_means)
        return cluster_centers[best_idx:best_idx+1]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        return y[:, np.newaxis]

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        self.n_evals = self.X.shape[0]
        self.best_point = self.X[np.argmin(self.y)]
        
        # Initialize ensemble
        for i in range(8):
            self.gp_ensemble[i] = self._initialize_gp(self.X, self.y, i)

        while self.n_evals < self.budget:
            # Fit ensemble less frequently
            self._fit_ensemble(self.X, self.y, self.n_evals)
            
            # Select diverse batch of 2 points (parallelizable)
            next_points = self._select_next_points(2)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)
            self.n_evals += next_points.shape[0]

            # Update trust region based on ensemble variance
            region_center = self.X[np.argmin(self.y)]
            ensemble_preds = [gp.predict(region_center[np.newaxis, :]) for gp in self.gp_ensemble]
            ensemble_std = np.std(ensemble_preds)
            
            # Adaptive radius: larger when ensemble disagreement is high
            if ensemble_std > 0.1:
                self.trust_radius *= 1.2
            else:
                self.trust_radius *= 0.8
            self.trust_radius = np.clip(self.trust_radius, 0.1, 5.0)
            self.best_point = self.X[np.argmin(self.y)]

        best_idx = np.argmin(self.y)
        return self.y[best_idx], self.X[best_idx]
