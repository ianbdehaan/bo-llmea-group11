from collections.abc import Callable
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
import torch
import torch.nn as nn
import torch.optim as optim

class DropoutBNNBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Initial design for BNN
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)

        class BNN(nn.Module):
            def __init__(self, dim):
                super(BNN, self).__init__()
                self.dropout1 = nn.Dropout(0.2)
                self.dropout2 = nn.Dropout(0.2)
                self.fc1 = nn.Linear(dim, 32)
                self.fc2 = nn.Linear(32, 1)

            def forward(self, x):
                x = self.dropout1(torch.relu(self.fc1(x)))
                x = self.dropout2(self.fc2(x))
                return x

        model = BNN(self.dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for _ in range(50):  # Reduced training epochs for efficiency
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return model

    def _acquisition_function(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        pred_samples = []
        with torch.no_grad():
            for _ in range(10):  # Dropout samples for uncertainty
                model.train()  # Enable dropout during prediction
                preds = self.model(X_tensor)
                pred_samples.append(preds.numpy().flatten())

        mu = np.mean(pred_samples, axis=0)
        sigma = np.std(pred_samples, axis=0)
        sigma = np.clip(sigma, 1e-9, None)
        f_min = np.min(self.y)
        z = (f_min - mu) / sigma
        ei = (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(z >= 0, ei, 0.0)  # EI is zero if predicted > current best
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(50)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-5:]]

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()

            bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
            res = minimize(
                acq_wrapper,
                x0, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

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
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
