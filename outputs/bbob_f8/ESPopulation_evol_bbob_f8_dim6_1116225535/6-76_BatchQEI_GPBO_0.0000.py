from collections.abc import Callable
from scipy.optimize import Bounds, minimize, LbfgsInvHessProduct
import numpy as np
from sklearn.gaussian_process.kernels import Matern
import torch
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.variational import VariationalStrategy
from gpytorch.lazy import LazyTensor
from gpytorch.distributions import MultivariateNormal

class BatchQEI_GPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 10)  # Ensure sufficient diversity
        self.best_x = None
        self.best_y = float('inf')
        self.batch_size = 5  # Dynamically adjusted in __call__
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.torch_device = torch.device("cpu")
        self.torch_dtype = torch.float32

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        class GPModel(ExactGP):
            def __init__(self, train_X, train_y, likelihood):
                super(GPModel, self).__init__(train_X, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(torch.tensor(X, dtype=self.torch_dtype), 
                        torch.tensor(y, dtype=self.torch_dtype), likelihood)
        model.train()
        likelihood.train()

        # Training loop
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(30):  # Few iterations for efficiency
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets).sum()
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        return model, likelihood

    def _select_next_batch(self):
        X_tensor = torch.tensor(self.X, dtype=self.torch_dtype)
        y_tensor = torch.tensor(self.y, dtype=self.torch_dtype)
        model, _ = self._fit_model(X_tensor, y_tensor)
        self.model = model

        def objective(batch):
            batch_tensor = torch.tensor(batch, dtype=self.torch_dtype, requires_grad=True)
            with torch.no_grad():
                posterior = model.posterior(batch_tensor)
                mean = posterior.mean
                std = posterior.stddev.clamp_min(1e-6)
                improvement = (mean - self.best_y).clamp_min(0)
                z = improvement / std
                ei = (improvement * torch.distributions.Normal(0, 1).cdf(z)) + (std * torch.distributions.Normal(0, 1).log_prob(z)).exp()
                ei = -ei.sum().item()
            return ei

        # Generate candidates via Sobol and optimize
        sobol_candidates = self._sample_in_trust_region(50)
        batch_candidates = []
        for candidate in sobol_candidates:
            res = minimize(
                fun=lambda x: objective(x.reshape(1, -1)),
                x0=candidate,
                method='L-BFGS-B',
                bounds=Bounds(self.bounds[0], self.bounds[1]),
                options={'maxiter': 20}
            )
            batch_candidates.append(res.x)
        batch_candidates = np.array(batch_candidates)

        # Select best batch via greedy
        batch_indices = np.argsort([objective(c.reshape(1, -1)) for c in batch_candidates])[:self.batch_size]
        return batch_candidates[batch_indices]

    def _sample_in_trust_region(self, n_points):
        if self.best_x is None:
            return self._sample_points(n_points)
        lower = np.clip(self.best_x - 1.0, self.bounds[0], self.bounds[1])
        upper = np.clip(self.best_x + 1.0, self.bounds[0], self.bounds[1])
        return self.rng.uniform(low=lower, high=upper, size=(n_points, self.dim))

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
        best_idx = np.argmin(self.y)
        self.best_x = self.X[best_idx]
        self.best_y = self.y[best_idx]
        self.batch_size = max(5, self.budget // 100)  # Adjust batch size based on budget

        while self.n_evals < self.budget:
            next_batch = self._select_next_batch()
            new_y = self._evaluate_points(func, next_batch)
            self._update_eval_points(next_batch, new_y)

            for i in range(len(new_y)):
                if new_y[i] < self.best_y:
                    self.best_y = new_y[i]
                    self.best_x = next_batch[i]
            # Adjust batch size for remaining budget
            self.batch_size = max(2, int(self.batch_size * 0.9))

        return self.best_y, self.best_x
