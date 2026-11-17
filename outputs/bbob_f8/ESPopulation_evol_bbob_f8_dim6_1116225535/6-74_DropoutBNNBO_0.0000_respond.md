# Thinking (reasoning of the model, not part of the answer)
Okay, I need to design a new Bayesian Optimization algorithm that's different from the existing ones. Let's look at the current algorithms and see what they're using.

The existing ones use Trust Regions with either GP or Random Forest models. The latest TrustRegionEnsembleThompsonBO uses Random Forest ensemble with Thompson Sampling. The main components there are Trust Region management, Latin Hypercube initialization, and Thompson Sampling with ensemble variance as the acquisition.

I need to come up with something different. Let's think about surrogate models. The current ones use GP (with RBF/Matern) or Random Forest. Maybe using a different surrogate model, like a Bayesian Neural Network (BNN) could be a good idea. However, BNNs can be computationally heavy, especially with small data. Wait, the problem constraints mention that each run should be under 120 seconds. So maybe BNN is too slow here. Alternative could be a simpler model, perhaps a neural network with Bayesian layers, but that might still be too slow.

Alternatively, what about using a different acquisition function? The current ones use Expected Improvement, LCB, Thompson Sampling. Maybe using a different approach like entropy search or predictive entropy search? But I'm not sure if that's feasible under the time constraints.

Another angle: instead of Trust Regions, maybe use a different exploration strategy. For example, using a particle swarm optimization approach but integrated with Bayesian methods. Hmm, but the user wants a Bayesian Optimization algorithm. Maybe a hybrid approach.

Wait, the problem mentions the Rosenbrock function, which has a long, narrow valley. The current algorithms use Trust Regions to focus search in promising areas. What if I use a different search strategy that doesn't rely on Trust Regions but uses a global model with adaptive exploration?

Another thought: using a surrogate model with a different kernel. The DynamicLCBGPBO uses anisotropic RBF kernel. Maybe using a different kernel, like a Matern kernel with different smoothness, but that's similar to existing GPs.

Alternatively, what about using a probabilistic model that's more suited for high-dimensional spaces or non-stationary functions? Or maybe combining different models.

Wait, another idea: using a Gaussian Process with a deep kernel. However, deep kernels might be complex and computationally expensive.

Alternatively, using a Bayesian Optimization algorithm that uses a different acquisition function optimization method. For example, instead of Sobol candidates and L-BFGS-B, using a covariance matrix adaptation evolutionary strategy (CMA-ES) for acquisition function optimization. But integrating CMA-ES into the acquisition function optimization might be too computationally heavy for the time constraints.

Hmm, maybe the key is to change the surrogate model and acquisition function. The existing TrustRegionEnsembleThompsonBO uses Random Forest and Thompson Sampling. What if instead of using the ensemble variance as the acquisition, we use the entropy of the predictions? Or use the ensemble's mean and variance in a different way?

Wait, another approach: using a Bayesian Neural Network as the surrogate model. For instance, using a simple neural network with dropout for uncertainty estimation. The dropout can simulate Bayesian inference. Then, using Expected Improvement based on the neural network's predictions. However, training a neural network might be too slow for small budgets. But since the budget is up to 600, perhaps it's manageable with a lightweight network.

But the user mentioned that the code should use libraries like numpy, scipy, scikit-learn, torch, gpytorch. So using PyTorch is allowed. Let me think about how to implement a Bayesian Neural Network using dropout as a surrogate.

The surrogate model could be a simple feedforward network with dropout layers. During training, the dropout is applied, and during prediction, multiple forward passes with dropout are made to get uncertainty estimates. The acquisition function could be Expected Improvement or Upper Confidence Bound based on the network's mean and variance.

But integrating that into the BO loop would require careful handling. Also, for the Rosenbrock function in 6 dimensions, the model might need to be trained efficiently. Let's outline the steps:

1. Initialize with Sobol or Latin Hypercube design.
2. Use a Bayesian Neural Network (with dropout) as the surrogate model.
3. Acquisition function could be Expected Improvement with a UCB-like approach using the model's mean and variance.
4. Optimize the acquisition function using a combination of random candidates and local optimization.

But I need to ensure that the code is efficient and doesn't take more than 120 seconds. Training a neural network might be too slow for small datasets. However, with a small network and few epochs, it might be feasible.

Another point: the current TrustRegionEnsembleThompsonBO uses Thompson Sampling with the ensemble's variance. Maybe instead of Thompson Sampling, using Expected Improvement but with a BNN's predictive distribution.

Alternatively, using a Variational Inference approach for the Bayesian Neural Network.

Wait, but in code, implementing a Bayesian Neural Network with dropout in PyTorch would be manageable. Let me sketch this.

The surrogate model would be a simple neural network with two hidden layers and dropout between them. The loss function would be the negative log-likelihood, assuming Gaussian noise. During training, we sample the dropout masks, and during prediction, we sample multiple times to get the mean and variance.

The acquisition function could then be Expected Improvement (EI) calculated based on the network's predictive mean and variance. Or another acquisition function like Upper Confidence Bound (UCB).

Now, the key components would be:

- Surrogate Model: Bayesian Neural Network with dropout for uncertainty estimation.
- Acquisition Function: Expected Improvement or Upper Confidence Bound using the network's predictive distribution.
- Acquisition Optimization: Hybrid approach of random candidates and local optimization.
- Initialization: Sobol sequence for better coverage.

But how does this differ from existing algorithms? The current ones use GPs or Random Forests. Using a BNN is a different surrogate model approach. The BNN can capture non-stationary patterns and complex relationships, which might be beneficial for the Rosenbrock function's elongated valley.

Additionally, the Trust Region approach in existing algorithms focuses the search locally. If I forgo the Trust Region and use a global model, but with better uncertainty estimation, maybe the algorithm can balance exploration and exploitation more effectively without the Trust Region.

So the new algorithm could be named DropoutBNGPBO (Bayesian Neural Network with Dropout for BO), but maybe something more descriptive.

But the problem says to name the class based on the characteristics with the template <characteristics>BO. So if the main idea is using a Bayesian Neural Network with dropout as the surrogate model, then maybe DropoutBNGPBO or BNNBO.

But the existing DynamicLCBGPBO uses GP. So the new algorithm uses a BNN as the surrogate model instead of a GP or Random Forest.

Let me structure this:

Description: A Bayesian Optimization algorithm that employs a Bayesian Neural Network with dropout layers as the surrogate model, combined with Expected Improvement acquisition and a hybrid acquisition function optimizer. The algorithm leverages the neural network's predictive uncertainty for robust exploration-exploitation trade-offs.

Justification: The use of a Bayesian Neural Network provides a flexible non-stationary surrogate capable of capturing complex patterns in the Rosenbrock landscape. Dropout during prediction simulates Bayesian inference, enabling uncertainty quantification without the computational burden of GP kernel learning. The hybrid acquisition function optimizer combines random candidates with gradient-based refinement to efficiently locate optima, while Expected Improvement balances exploration and exploitation based on the BNN's predictions.

Now, implementing this in code. Let's outline the code steps.

First, in the __init__ method, set up the budget, dim, bounds, etc. Then, in the _sample_points method, use Sobol or Latin Hypercube for initial design.

The surrogate model would be a PyTorch model. Let's define a simple neural network with dropout. The _fit_model method would train this network using the collected data.

The acquisition function would compute Expected Improvement using the BNN's predictive mean and variance. For Expected Improvement, we need the current best value, the mean, the variance, and then compute EI.

The _select_next_points function would generate candidates via Sobol, evaluate the acquisition function, and perform local optimization.

But in code, the BNN training needs to be efficient. Also, during prediction, we sample multiple times (e.g., 10 samples) to compute the mean and variance.

Let me outline the code structure.

Import necessary libraries, including torch.

In the _fit_model function, define the BNN model with dropout. Then, train it using the collected X and y.

In the _acquisition_function, use Expected Improvement.

But how to compute Expected Improvement with the BNN's mean and variance?

The Expected Improvement formula requires the current best (f_min), the mean (mu), and the standard deviation (sigma) of the predictive distribution.

The EI at a point x is:

EI = (f_min - mu) * norm.cdf((f_min - mu)/sigma) + sigma * norm.pdf((f_min - mu)/sigma)

But here, mu and sigma are the BNN's predictions.

But for the BNN, during prediction, we take multiple samples (with dropout) to get the mean and variance. Let's say we take 10 samples, compute the mean as the average of the samples, and the variance as the variance of the samples.

So in code, during the _acquisition_function, for each candidate x, we sample the BNN multiple times, compute the mean and variance, then compute the EI.

But this could be computationally expensive if done for many candidates. However, given the constraints, perhaps with a small BNN and few samples, it's manageable.

Putting this all together, the new algorithm would be a Bayesian Optimization using BNN with dropout as the surrogate model, Expected Improvement as the acquisition function, and a hybrid acquisition function optimizer.

Now, to code this.

The code will have to include the BNN model, which is a PyTorch module. Training it will involve defining a loss function (e.g., negative log-likelihood), optimizer, and training loop.

But for efficiency, the training should be kept simple, perhaps with a few epochs.

Also, the initial design will use Sobol or Latin Hypercube to get a good spread.

The _evaluate_points method remains similar to the existing code.

Now, let's write the code.

The class name would be something like DropoutBNNBO.

But need to check the template. The user says to name it based on the characteristics with template <characteristics>BO. So since it's using a Bayesian Neural Network with dropout, maybe BNNBO, or DropoutBNNBO.

The code will look like this:

Import torch, define the BNN model with dropout.

In the __init__ method, set up the budget, dim, etc.

The _fit_model method will create the BNN model, train it on the data.

The _acquisition_function will compute Expected Improvement using the BNN's predictions.

The _select_next_points will generate candidates, evaluate the acquisition function, and perform local optimization.

Now, code steps:

In the __init__:

- Initialize the budget, dim, etc.
- Set up the random number generator.

Sample initial points with Sobol.

Fit the BNN model.

In the _fit_model method, create a neural network with PyTorch.

The acquisition function:

- For each candidate x, pass it through the BNN multiple times (e.g., 10 samples) to get predictions. Compute the mean and variance of these samples.

Then compute EI using the formula.

Optimize the acquisition function using a hybrid approach: generate random candidates (Sobol), evaluate their acquisition value, then perform local optimization from the top candidates.

Now, code.

But need to make sure that the BNN training is efficient and doesn't take too long.

Also, the code needs to be compatible with the evaluator's setup, which calls np.random.seed and torch.manual_seed before each run. So all random generators must be properly seeded.

In code:

from torch import nn, optim
import torch.distributions as dist

Then define the BNN class.

But in the existing code, the _fit_model is part of the algorithm class. So integrating the BNN into that.

But for the purposes of the code, the _fit_model function will create a PyTorch model, train it using the collected X and y data.

So here's an outline:

class DropoutBNNBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim + 1  # Maybe larger initial design for BNN
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        # Similar to previous code, Latin Hypercube or Sobol
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)

        # Define BNN model
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

        # Training loop
        for _ in range(100):  # Maybe 100 epochs
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return model

    def _acquisition_function(self, X):
        # X is a numpy array of candidates
        # Convert to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)

        # Predict with BNN and compute mean and variance
        pred_samples = []
        with torch.no_grad():
            for _ in range(10):  # Sample 10 times with dropout
                model.train()  # Enable dropout during prediction
                preds = self.model(X_tensor)
                pred_samples.append(preds.numpy().flatten())

        pred_samples = np.array(pred_samples)
        mu = np.mean(pred_samples, axis=0)
        sigma = np.std(pred_samples, axis=0)
        sigma = np.clip(sigma, 1e-9, None)

        # Compute Expected Improvement
        f_min = np.min(self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (f_min - mu) / sigma
            ei = (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(z >= 0, ei, 0.0)  # EI is zero if mu > f_min

        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Generate candidates via Sobol
        candidates = self._sample_points(50)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-5:]]  # Top 5 candidates

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
        # Same as before
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
        # Same as before
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initialize with Sobol design
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Train initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            # Update best
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]

            # Refit model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x

But wait, in the _acquisition_function, when we call model.train(), that's necessary to enable dropout during prediction. But during training, we also use model.train(), which is okay.

Another thing: the initial design for the BNN might need more points. The current n_init is set to 2*dim + 1. The existing TrustRegionEnsembleThompsonBO uses dim + 1. Maybe increasing it to 2*dim or similar for better initial data for the BNN.

Also, the BNN model has dropout layers. During prediction, we need to enable dropout by calling model.train(), then perform multiple forward passes to get the predictive distribution. But in PyTorch, when model.train() is called, dropout is applied. So during the acquisition function, for each candidate point, the model is in training mode, and we sample 10 times with dropout enabled to get the variance.

This should give an estimate of the predictive mean and variance.

Now, the key differences from the existing algorithms:

- Surrogate model: Uses a Bayesian Neural Network with dropout instead of GP or Random Forest.
- Acquisition function: Uses Expected Improvement based on the BNN's predictive distribution.
- Acquisition optimization: Hybrid approach of Sobol candidates and L-BFGS-B refinement.

The existing TrustRegionEnsembleThompsonBO uses Thompson Sampling with Random Forest ensemble, while this new algorithm uses Expected Improvement with a BNN. The BNN provides a more flexible model, which may be better at capturing the elongated valley of the Rosenbrock function. The hybrid acquisition function optimization should be efficient enough.

Potential issues: The BNN training might be computationally expensive for small datasets. However, with a small network and few epochs, it might be manageable. Also, the code needs to make sure that all random generators are properly seeded, including PyTorch.

In the code, the BNN is trained for 100 epochs each time a new model is fitted. However, with a small number of points, this could overfit. Maybe reduce the number of epochs to 50 or so.

Another point: the acquisition function uses Expected Improvement, which is a global acquisition function. Unlike the Trust Region approach in previous algorithms, this algorithm doesn't restrict the search to a local region. However, the BNN's uncertainty estimation might still allow for efficient exploration without Trust Regions.

This should be a novel approach compared to the existing algorithms in the population.


# Answer
# Description 
**DropoutBNNBO** is a Bayesian Optimization algorithm that employs a Bayesian Neural Network (BNN) with dropout layers as the surrogate model, combined with Expected Improvement (EI) acquisition and a hybrid acquisition function optimizer. The algorithm leverages the BNN's predictive uncertainty to balance exploration and exploitation, while the hybrid optimizer combines random candidates with gradient-based refinement for efficient acquisition function maximization. Key innovations include:  
- **Bayesian Neural Network Surrogate**: Utilizes dropout for approximate Bayesian inference to capture complex patterns in the Rosenbrock landscape.  
- **Expected Improvement Acquisition**: Computes EI using the BNN's predictive mean and variance for robust exploration.  
- **Hybrid Acquisition Optimization**: Balances global sampling (Sobol) and local refinement (L-BFGS-B) to efficiently locate optima.  

# Justification 
The DropoutBNNBO algorithm differs from existing algorithms by replacing Gaussian Processes and Random Forests with a BNN for uncertainty modeling. The BNN's dropout-based Bayesian inference provides a more flexible and scalable surrogate for high-dimensional and non-stationary functions like Rosenbrock. Expected Improvement ensures principled exploration-exploitation trade-offs, while the hybrid optimizer reduces computational overhead. The algorithm avoids trust regions entirely, relying on global modeling for adaptability. Key improvements over TrustRegionEnsembleThompsonBO:  
1. **Surrogate Model**: BNN replaces Random Forest for better uncertainty quantification and non-stationary pattern modeling.  
2. **Acquisition Function**: EI replaces Thompson Sampling, using BNN's predictive distribution for robust optimization.  
3. **Search Strategy**: Hybrid optimization (global + local) ensures scalability without trust region constraints.  

# Code 
```python
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
```
## Error
 Traceback (most recent call last):
  File "<DropoutBNNBO>", line 131, in __call__
 131->             next_point = self._select_next_points(1)
  File "<DropoutBNNBO>", line 77, in _select_next_points
  77->         acq_values = self._acquisition_function(candidates)
  File "<DropoutBNNBO>", line 62, in _acquisition_function
  60 |         with torch.no_grad():
  61 |             for _ in range(10):  # Dropout samples for uncertainty
  62->                 model.train()  # Enable dropout during prediction
  63 |                 preds = self.model(X_tensor)
  64 |                 pred_samples.append(preds.numpy().flatten())
NameError: name 'model' is not defined. Did you mean: 'self.model'?
