from collections.abc import Callable
from scipy.optimize import Bounds, minimize
from scipy.stats import qmc, norm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import check_random_state

class BayesianNeuralNetworkEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(2 * dim, 10)  # Ensure at least 10 initial points
        self.best_x = None
        self.best_y = float('inf')
        self.random_state = check_random_state(None)
        self.rng = np.random.default_rng(self.random_state.randint(2**31 - 1))
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(self.random_state.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    class BNN(nn.Module):
        def __init__(self, dim, hidden=64, dropout=0.1):
            super().__init__()
            self.dim = dim
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1)
            )
        
        def forward(self, x):
            return self.net(x)

    def _train_bnn(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=False).view(-1, 1)
        
        model = self.BNN(self.dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        for _ in range(100):  # Fixed number of epochs for efficiency
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model

    def _predict_bnn(self, model, X, n_samples=10):
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        predictions = []
        model.train()  # Enable dropout for MC sampling
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(X_tensor).numpy()
                predictions.append(pred)
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0).flatten()
        std = predictions.std(axis=0).flatten()
        return mean, std

    def _select_next_point(self, model):
        candidates = self._sample_points(200)
        mean, std = self._predict_bnn(model, candidates)
        
        # Expected Improvement acquisition function
        improvement = mean - self.best_y
        z = improvement / (std + 1e-9)
        ei = (improvement * norm.cdf(z)) + (std * norm.pdf(z))
        ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
        
        # Local refinement
        best_candidate = candidates[ei.argmax()].reshape(1, -1)
        bounds = Bounds(self.bounds[0], self.bounds[1])
        
        def objective(x):
            x = x.reshape(1, -1)
            mu, sigma = self._predict_bnn(model, x, n_samples=10)
            sigma = np.clip(sigma, 1e-9, None)
            improvement = mu - self.best_y
            z = improvement / sigma
            ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
            ei = np.nan_to_num(ei, neginf=0.0, posinf=0.0)
            return -ei.item()
        
        result = minimize(objective, best_candidate, method='L-BFGS-B', bounds=bounds, tol=1e-6)
        if result.success:
            next_point = result.x.reshape(1, -1)
        else:
            next_point = best_candidate
        return next_point

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
        
        while self.n_evals < self.budget:
            model = self._train_bnn(self.X, self.y)
            next_point = self._select_next_point(model)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            
            if new_y[0] < self.best_y:
                self.best_y = new_y[0]
                self.best_x = next_point[0]
        
        return self.best_y, self.best_x
