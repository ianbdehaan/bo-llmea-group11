from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

class PortfolioForestEIComboBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 10)  # Ensure sufficient initial points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_design = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))

    def _sample_points(self, n_points):
        samples = self.lhs_design.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Use random forest with variance-based uncertainty estimation
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1
        )
        model.fit(X, y.ravel())
        return model

    def _acquisition_function(self, X, model):
        # Expected Improvement
        mean = model.predict(X).ravel()
        var = np.var([tree.predict(X) for tree in model.estimators_], axis=0)  # Ensemble variance
        std = np.sqrt(np.clip(var, 1e-8, None))
        best_y = np.min(self.y)
        improvement = best_y - mean  # Minimization problem
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0
        
        # Thompson Sampling: sample a function from the posterior
        thompson_samples = np.array([tree.predict(X) for tree in model.estimators_])
        thompson_mean = np.mean(thompson_samples, axis=0)
        
        # Portfolio: 50% EI + 50% Thompson Sampling
        combined_acq = 0.5 * ei + 0.5 * (-thompson_mean)  # Minimize TS samples
        return combined_acq

    def _optimize_acquisition(self, model):
        # Generate 50 Sobol candidates
        candidates_sobol = self._sample_points(50)
        acq_sobol = self._acquisition_function(candidates_sobol, model)
        
        # Local optimization from top 3 best points
        top3_indices = np.argsort(acq_sobol)[-3:]
        candidates_local = candidates_sobol[top3_indices]
        candidates_local_opt = []
        
        for x0 in candidates_local:
            def _negative_acq(x):
                return -self._acquisition_function(x.reshape(1, -1), model)
            
            result = scipy.optimize.minimize(
                _negative_acq, 
                x0=x0, 
                bounds=self.bounds.T,
                method='L-BFGS-B',
                options={'maxiter': 20}
            )
            if result.success:
                candidates_local_opt.append(result.x)
        
        candidates_local_opt = np.array(candidates_local_opt)
        all_candidates = np.vstack([candidates_sobol, candidates_local_opt])
        acq_vals = self._acquisition_function(all_candidates, model)
        best_idx = np.argmax(acq_vals)
        return all_candidates[best_idx].reshape(1, -1)

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
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._optimize_acquisition(self.model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
