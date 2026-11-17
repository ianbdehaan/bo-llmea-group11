# Thinking (reasoning of the model, not part of the answer)
Okay, let's see. The user wants me to design a new Bayesian Optimization algorithm that's different from the existing ones like DynamicLCBBO, MaternLCBBO, ATRBO, and TrustRegionEI_BO. The existing solutions use various techniques like LCB, EI, trust regions, Sobol sequences, and Matérn kernels.

The main goal is to create something novel. Let's think about what components are common. Most use Gaussian Processes with Matérn kernels, acquisition functions like EI or LCB, and initial designs like Sobol. To be different, maybe change the surrogate model or acquisition function. Also, the existing ones use exploration strategies that change over time, but perhaps there's a different way.

Hmm, what if I use a different surrogate model? Maybe a random forest instead of a GP? But the user mentioned that GPs are allowed, and maybe they want to stick with GPs but with a different approach. Alternatively, maybe using a log-transformed GP as mentioned in the options. The current DynamicLCBBO uses standard GP.

Wait, in the options, there's an example of using a GP on log-transformed y. The Rosenbrock function might benefit from transforming the target values to stabilize variance, especially since the function's output can vary widely. So maybe using a log-transformed GP for the surrogate model. That could help in cases where the target variable has a skewed distribution.

For the acquisition function, maybe instead of LCB or EI, use a different one. Thompson Sampling? Or maybe a portfolio approach that switches between acquisition functions. But Thompson Sampling might be computationally more expensive. Alternatively, use a variant of Expected Improvement that's modified for better exploration.

Another idea: using a different acquisition function optimizer. The existing ones use random sampling plus L-BFGS-B. What if we use a different optimizer, like differential evolution for the acquisition function? Or maybe a batch selection method that picks diverse candidates to speed up convergence.

Wait, the problem is the Rosenbrock function which has a long, narrow valley. So the algorithm needs to explore effectively in that region. Maybe using an acquisition function that encourages movement along the valley. For example, using a gradient-based approach, but since it's a black box, we can't get gradients. So perhaps the acquisition function could be designed to prefer points in the direction of the best current point.

Alternatively, use a trust region approach but with a different expansion mechanism. The existing TrustRegionEI_BO uses dynamic adjustment. But maybe combine trust regions with another strategy. However, the user wants something different from all current algorithms.

Let me think about the components. The existing DynamicLCBBO uses a dynamic LCB with decaying beta. So maybe a new algorithm that uses a different exploration strategy. For example, instead of decaying over time, adjust based on the uncertainty in the model. Or use a multi-fidelity approach, but the problem is single-fidelity.

Another angle: the acquisition function optimization. The current code uses random candidates plus L-BFGS-B. Maybe using a more efficient method for optimizing the acquisition function. For example, using a quasi-Monte Carlo approach for generating candidates, or using a different optimization strategy like Covariance Matrix Adaptation Evolution Strategy (CMA-ES) but that might be too slow.

Wait, the user mentioned that the algorithm must be computationally efficient, with cost growing moderately. So using a complex optimization method might be too slow. Maybe sticking with some random sampling but in a more adaptive way.

Alternatively, using a batch Bayesian Optimization approach. But the existing algorithms don't use batches except maybe in the acquisition function optimization. However, the current code uses sequential selection.

Another idea: combining different acquisition functions. For example, a portfolio approach where different acquisition functions are used in different phases of the optimization. But how to implement that without complicating the code.

Let me think about the surrogate model again. The current ones use Matérn kernel with ARD. What if we use a different kernel? Like a combination of kernels, but that might complicate hyperparameter tuning. Or maybe use a simpler kernel like RBF but with a different approach for handling the length scales.

Wait, the problem requires that the algorithm is suitable for the Rosenbrock function. The Rosenbrock function has a long, narrow valley. The existing algorithms use anisotropic kernels (ARD) which is good for such scenarios. Maybe the new algorithm can use a different way of adapting the kernel based on the current best point.

Alternatively, using a GP that is updated only in the vicinity of the best point (local GP). But the existing TrustRegion algorithms do something similar. Hmm.

Wait, the user mentioned that the algorithm must be efficient and work for any budget. So maybe the new approach uses a different acquisition function. Let's consider the Lower Confidence Bound (LCB) but with a different exploration parameter. The DynamicLCBBO uses a decaying beta. What if instead of decaying over time, we base it on the current model's uncertainty in the best region? For example, higher exploration where the model is uncertain around the best point.

Alternatively, using a different acquisition function like Max Value Entropy Search (MES) but that's more complex. Or Knowledge Gradient. But these might be too computationally intensive for the time constraints.

Another thought: using a different initial design. The current code uses Sobol. What if we use a different initial design, like a maximin Latin hypercube? But the user might prefer to use Sobol for space-filling.

Wait, the existing DynamicLCBBO already uses Sobol for initial design. To be different, maybe using a different initial design for the first few points and then switching. But not sure.

Let me think about the surrogate model again. The current algorithms use a standard GP. Maybe using a GP with a different transformation of the target variable. For example, using a log transformation on y to stabilize the variance. This could help in cases where the function has a skewed distribution. The Rosenbrock function's minimum is 0, but as you move away, the values increase quadratically. Taking the log might not be suitable because when y is 0, log is undefined. Hmm. Alternatively, use a shifted log transformation, like log(y + 1). But maybe this isn't necessary. Alternatively, use a GP that's designed for heteroscedastic noise, but that's more complex.

Alternatively, use a GP with a non-stationary kernel. But that's again more complex. The user wants computational efficiency, so maybe avoiding non-stationary kernels.

Hmm. Another angle: acquisition function optimization. The current code uses 50 random candidates and selects the top 5 for local optimization. Maybe increasing the number of candidates but optimizing more efficiently. For example, using a different number of random starts, or a different optimization method. But the user requires computational efficiency.

Wait, the user mentioned that the algorithm must be efficient. The current DynamicLCBBO uses L-BFGS-B for local optimization. Maybe switching to a different optimization method for the acquisition function, like differential evolution, but that's more computationally heavy. However, the code must handle 600 evaluations, so the acquisition function optimization can't be too slow.

Alternatively, use a random sampling approach without local optimization. For example, randomly generate a large number of candidates and select the one with the best acquisition value. But that might miss the optimal points.

Alternatively, use a different strategy for generating the candidates. For example, using a quasi-random sequence like Halton or Sobol for the candidates. The current code already uses Sobol for the initial design, but for the acquisition function candidates, maybe using a different method.

Wait, perhaps the new algorithm uses a different strategy for the acquisition function optimization. For example, generating candidates around the best current point to encourage exploitation, while still allowing some exploration.

Let me try to outline the new algorithm. Let's focus on a different surrogate model. The existing ones use GP with Matérn. What if we use a random forest as the surrogate model? The problem with random forests is that they don't naturally provide uncertainty estimates, but we can use the variance of the tree predictions as a proxy for uncertainty. This could be a different approach compared to GP-based BO. However, the user mentioned that the algorithm must be efficient, and random forests are generally faster in prediction but may require more data.

Wait, the Rosenbrock function has a very specific structure. Maybe using a random forest could capture the structure, but the uncertainty estimation might not be as reliable as GPs. However, since the problem is a black-box, maybe it's worth exploring. But the user allows using scikit-learn, which does have random forests.

But given that the existing algorithms are using GPs, maybe the new algorithm can be based on a different surrogate model. However, the user's code uses sklearn's GP. Let me check the code again.

In the existing code, the surrogate is a GP with Matérn kernel. So to be different, maybe using a different kernel. For example, using a RBF kernel with automatic relevance determination. But that's similar to Matérn. Alternatively, using a periodic kernel, but that's not suitable for Rosenbrock.

Alternatively, using a mixture of kernels. But again, that might complicate the hyperparameter tuning.

Wait, another idea: using an ensemble of GPs. For example, multiple GPs with different kernels and combining their predictions. However, that would increase the computational cost, which the user wants to avoid.

Alternatively, using a simplified GP model. For example, not using ARD, but a fixed kernel. But that might not be as effective.

Hmm. Maybe the problem is in the acquisition function. The existing DynamicLCBBO uses a decaying beta. What if we use a fixed beta but with a different acquisition function? For example, using a combination of Expected Improvement and LCB. But that's a portfolio approach.

Alternatively, using a Thompson Sampling approach. Thompson Sampling samples a function from the GP posterior and then optimizes it. But how to implement that efficiently. Thompson Sampling can be effective in multi-armed bandit problems, but for high-dimensional spaces, it's challenging. However, since the problem is 6D, maybe it's manageable. However, optimizing a sampled function might be time-consuming. But perhaps using a simple optimizer for the sampled function.

Wait, the user allows using torch and gpytorch. Maybe using a more advanced GP implementation with those libraries. But the current code uses sklearn's GP. Let's check.

The user says that the code must use numpy, scipy, scikit-learn, torch, gpytorch. So using torch or gpytorch is allowed. So perhaps using a different GP implementation from torch that's more efficient or uses a different approach.

But the code must be in Python, and the user's existing code uses scikit-learn's GP. So perhaps using a different acquisition function optimization method. For example, using a Bayesian quadrature approach, but that might be too slow.

Alternatively, using a different acquisition function, like Maximum Mean Discrepancy (MMD) based, but that's more complex.

Wait, what about using a different strategy for balancing exploration and exploitation? For example, using entropy search or predictive entropy search. But again, these are more computationally intensive.

Another angle: using batch Bayesian Optimization. The current code selects one point at a time. Maybe selecting multiple points in a batch to speed up the evaluation. But the user's problem has each run taking 120 seconds, and the code must be efficient. So, for example, selecting a small batch each time could reduce the number of iterations and thus computational time. However, the user's code uses sequential selection. So using a batch approach could be a difference.

But how to implement a batch acquisition function. For example, using q-EI or q-LCB which selects a batch of points that are diverse. But implementing q-EI requires using a batched acquisition function, which might be complex. The user's code uses a sequential approach.

Given that the user's code is constrained by time and computational cost, perhaps using a simplified batch approach. For example, in each iteration, select multiple points that are diverse, but not necessarily optimizing a complex batch acquisition function. For example, selecting a few candidates in different regions of the space.

Alternatively, using a multi-point acquisition function that selects several points each iteration. However, this requires a more sophisticated approach.

Hmm. Let me think about a concrete example. Let's create an algorithm that uses a GP with a different approach to the acquisition function. For example, instead of LCB, use an Expected Improvement (EI) acquisition function with a dynamic exploration parameter. But the existing MaternLCBBO uses EI. Wait, no, the existing DynamicLCBBO uses LCB. So if we use EI but with a different strategy for the parameter.

But the MaternLCBBO's algorithm uses EI with a decaying exploration parameter. The user's code for DynamicLCBBO uses LCB with decaying beta. So to be different, perhaps using a different form of EI. For example, using a modified EI that takes into account the uncertainty around the best point.

Alternatively, use a Thompson Sampling approach. Let's think about that. Thompson Sampling for BO would sample a function from the GP posterior and then find the maximum of that sampled function. Since the Rosenbrock function's minimum is in a specific location, maybe this approach could help in exploring around the best point.

But implementing Thompson Sampling would require sampling from the GP's posterior. For a Gaussian Process, this is straightforward. The steps would be:

1. Fit the GP to the current data.
2. Sample a function from the GP's posterior.
3. Find the minimum of that sampled function (since it's a black-box function, this would require optimization).
4. Evaluate that point.

But optimizing the sampled function (which is a noisy estimate) could be time-consuming. However, using a simple optimizer like L-BFGS-B for each sampled function might be feasible.

This approach would differ from the existing algorithms. The existing ones use deterministic acquisition functions like LCB or EI. Thompson Sampling is a probabilistic approach. This could lead to different exploration behaviors, possibly adapting better to the specific structure of the Rosenbrock function.

So, the new algorithm could be called ThompsonSamplingBO. The main components would be:

- Initial design using Sobol sequences.
- Surrogate model: Gaussian Process with Matérn kernel (similar to existing ones but the acquisition function is different).
- Acquisition function: Thompson Sampling. For each iteration, sample a function from the GP posterior and optimize it to find the next point.
- Acquisition function optimization: Optimize the sampled function using multiple random restarts and L-BFGS-B.

But let's think about computational efficiency. Sampling from the GP posterior is O(n^3) for the Cholesky decomposition, which might be expensive for large budgets, but given that the maximum budget is 600, it's manageable. Also, each iteration requires optimizing a sampled function, which might be more computationally intensive than evaluating a deterministic acquisition function. However, the code must be efficient enough to run within 120 seconds. Let's test this in the code.

The code would need to generate a sample from the GP's posterior. However, in practice, the GP's posterior is a multivariate normal distribution. To sample a function at a given set of points, we need to compute the mean and covariance matrix. But for optimization, we need to sample a function, which is defined over the entire space. However, we can't do that directly. Instead, we can generate a finite set of points and sample the function at those points, but that's not feasible for high dimensions.

Wait, but in practice, when using Thompson Sampling for BO, the method is to sample a function from the GP posterior and then find its minimum. However, since the function is continuous, we can't sample it everywhere. So, the approach is to approximate it by generating a large number of candidate points, evaluating the sampled function at these points, and then selecting the minimum. But this is similar to random search, which is not efficient. Alternatively, use optimization to find the minimum of the sampled function. 

But how to do that? The sampled function is a realization of the GP, which is a Gaussian process. So, for any finite set of points, the sampled function is a multivariate normal distribution. The maximum (or minimum) of this function is what we're after. However, optimizing this requires an optimization routine that can find the extrema of the sampled function.

Alternatively, since the sampled function is a Gaussian process, perhaps the maximum can be approximated using some properties. But in practice, it's usually done by optimizing the sampled function using a local optimizer. For example, generate a random set of candidate points, evaluate the sampled function at these points, and then optimize using L-BFGS-B starting from the best candidate.

But this approach could be computationally expensive. However, given the constraints, maybe it's feasible. Let's outline the steps for the algorithm:

1. Initial design: Generate Sobol points.
2. Fit a GP model to the data.
3. For each iteration:
   a. Sample a function from the GP posterior.
   b. Optimize this sampled function using multiple random restarts and a local optimizer (like L-BFGS-B).
   c. Evaluate the best point from the optimization.
   d. Update the model.

This approach would differ from existing algorithms in the acquisition function (Thompson Sampling vs. LCB/EI) and the strategy for selecting the next point. The key components would be the use of Thompson Sampling to balance exploration and exploitation, which can adaptively explore areas where the model is uncertain.

Now, to implement this in code. The code would need to:

- In the _select_next_points method, generate a sampled function.
- Optimize this function.

But how to sample from the GP posterior. For a Gaussian Process, the sampled function can be evaluated at any point. The mean and covariance matrix of the GP give the distribution. However, for optimization, we need to evaluate the sampled function at a candidate point. So, during the acquisition function optimization, for each candidate point, evaluate the mean and covariance of the GP at that point, then sample a value from the normal distribution defined by the mean and the diagonal of the covariance matrix. 

Wait, no. The GP's posterior is a multivariate Gaussian distribution over the function values at the observed points. To sample a function, you'd need to generate a realization of the function at the observed points and then interpolate. However, in practice, the sampled function can be approximated by evaluating the GP at a large set of candidate points and sampling the function at those points. But this is computationally intensive.

Alternatively, during the acquisition function optimization, for each candidate point, we can sample a value from the posterior distribution at that point. Then, use those sampled values as the function to optimize. But this approach is not the same as sampling the entire function, but it's an approximation.

Wait, but Thompson Sampling in BO typically involves sampling a function from the posterior and then optimizing it. However, in practice, due to computational constraints, it's often approximated by generating samples at a set of candidate points. But this might not be feasible for high-dimensional spaces. 

Alternatively, for the sampled function, during optimization, we can generate a single sample for each candidate point during the optimization process. For example, when optimizing the sampled function, for a given candidate x, evaluate the posterior mean mu(x) and standard deviation sigma(x), then sample a value from N(mu(x), sigma(x)). The optimizer would then maximize this sampled value. 

But this is not the same as sampling a full function from the GP's posterior. However, it's a practical approximation. In this case, the acquisition function becomes the sampled value from the GP at each candidate point. This approach is known as "sampled value" Thompson Sampling. 

So, the code can be modified to implement this approach. For each candidate x, sample a value from the GP's posterior distribution at x, and then find the maximum of this sampled value. 

This approach is computationally feasible. The steps for _select_next_points would be:

- Generate a set of random candidates.
- For each candidate, sample a value from the GP's posterior at x.
- Optimize the sampled values to find the maximum.

But how to implement this in code. For each candidate, the GP's posterior distribution is normal with mean mu and variance sigma^2. So, we can generate a random sample from this distribution. 

In code terms, the acquisition function for Thompson Sampling would be:

def _acquisition_function(self, X):
    mu, sigma = self.model.predict(X, return_std=True)
    samples = mu + sigma * self.rng.normal(size=mu.shape)
    return samples

But in the optimization step, we want to maximize this sampled value. 

So, the _select_next_points method would generate candidates, evaluate the acquisition function (the sampled values), and then optimize them. 

This approach is different from the existing algorithms because it uses a probabilistic sampling approach instead of a deterministic acquisition function. 

Now, considering the computational cost. Generating the candidates and optimizing them using L-BFGS-B is similar to the existing code, but the acquisition function involves sampling from the normal distribution. This should be computationally efficient. 

Let's outline the code changes needed. 

In the existing DynamicLCBBO code, the _acquisition_function computes mu and sigma and returns mu - beta*sigma. For Thompson Sampling, it would sample from N(mu, sigma^2) and return those samples. 

The _select_next_points method would then select the point with the highest sampled value. However, since the sampled values are random, we might need to generate multiple samples per iteration to find a good point. 

But in the code, for each iteration, we can generate one sample, optimize it, and select the point. Alternatively, generate multiple samples and select the best among them. 

But to keep it efficient, let's generate one sample and optimize it. 

So, the new algorithm would be called ThompsonSamplingBO. 

Now, let's think about the code. The key changes would be in the _acquisition_function and _select_next_points methods. 

In the _acquisition_function, instead of returning mu - beta*sigma, return samples from N(mu, sigma^2). 

In the _select_next_points, generate candidates, evaluate the acquisition function (the sampled values), and then optimize those candidates with L-BFGS-B to find the maximum. 

But in the code, the _select_next_points currently uses the top 5 candidates for local optimization. For Thompson Sampling, perhaps we can just optimize the sampled function by using the same approach but with the sampled values. 

However, since the sampled values are random, maybe generating more candidates would help in finding a better maximum. 

Let's adjust the code accordingly. 

Another consideration is the initial design. The existing code uses Sobol sequences, which is good for space-filling. The new algorithm can retain that. 

For the surrogate model, we can continue using the Matérn kernel with ARD. 

Now, the main difference is the acquisition function being Thompson Sampling. 

The name would be ThompsonSamplingBO. 

Now, let's write the code. 

In the _acquisition_function method, compute mu and sigma, then generate samples. For each candidate, sample a value. 

But in the code, the acquisition function is used to select candidates. For example, in the current code, the candidates are generated, and the acquisition function is evaluated for each. Then, the top candidates are selected for local optimization. 

In the Thompson Sampling case, for each candidate, the sampled value is the acquisition value. 

But since each sample is random, perhaps generating multiple candidates and selecting the ones with the highest sampled values. 

However, the current code uses 50 candidates, takes the top 5, and optimizes them. For Thompson Sampling, perhaps the same approach is used, but the acquisition function is the sampled value. 

But the sampled values are different each time. 

In code terms, the _acquisition_function method would generate the samples, and then the _select_next_points method would proceed as before, but with the acquisition function being the sampled values. 

Thus, the code would look like this:

In the _acquisition_function:

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

Then, in _select_next_points, the candidates are evaluated for these samples. 

But in the original code, the acquisition function is minimized (since for LCB, lower values are better). In Thompson Sampling, we want to maximize the sampled value. Therefore, in the _select_next_points, the code would select the candidate with the highest acquisition value. 

In the current code, for the LCB, the acquisition function is mu - beta*sigma. The code sorts in ascending order, taking the first few for optimization. 

In the Thompson Sampling case, the acquisition values are the samples. We need to take the maximum. 

Thus, in the _select_next_points method, the line:

    idxs = np.argsort(acq_values, axis=0)[:, 0]

would change to:

    idxs = np.argsort(-acq_values, axis=0)[:, 0]

to sort in descending order. 

Alternatively, sort the acquisition values in descending order. 

So, the modified code would have:

    idxs = np.argsort(acq_values, axis=0)[:, 0][::-1]

But in the code, the top 5 candidates are selected for local optimization. 

Thus, the _select_next_points method remains largely the same, except the acquisition function is now sampled from the GP. 

This would be a significant change from the existing algorithms, which use deterministic acquisition functions. 

Now, let's think about the justification. 

The ThompsonSamplingBO algorithm uses Thompson Sampling for acquisition function optimization, which introduces a probabilistic approach to balancing exploration and exploitation. By sampling from the GP's posterior, it adaptively explores areas where the model is uncertain while exploiting areas with high predicted performance. This approach can lead to more efficient exploration of the Rosenbrock function's elongated valley, especially as the algorithm converges to the global minimum. 

The key components are:

- **Initial design using Sobol sequences** for space-filling sampling.
- **Gaussian Process surrogate model with Matérn-ARD kernel** to capture function structure.
- **Thompson Sampling acquisition function** that samples a function from the GP's posterior and optimizes it, promoting adaptive exploration.
- **Hybrid optimization approach** with random candidates and local search (L-BFGS-B) to find optimal points for the sampled function.

This approach differs from the existing algorithms by using a probabilistic sampling strategy instead of deterministic acquisition functions, potentially leading to better exploration in complex landscapes like the Rosenbrock function.


# Answer
# Description  
ThompsonSamplingBO is a Bayesian Optimization algorithm that utilizes Thompson Sampling for acquisition function optimization. It combines an anisotropic Matérn kernel Gaussian Process (GP) as a surrogate model with a dynamic sampling strategy from the posterior. The algorithm employs Sobol sequences for initial space-filling sampling and optimizes sampled functions from the GP posterior to balance exploration and exploitation. By sampling a function realization from the GP and optimizing it, the algorithm adapts to the Rosenbrock function's elongated valley structure, promoting efficient exploration while maintaining computational efficiency through random restarts and L-BFGS-B local optimization.

# Justification  
The core novelty of ThompsonSamplingBO lies in its **probabilistic acquisition strategy**:  
1. **Thompson Sampling (TS)**: Instead of deterministic acquisition functions like LCB or EI, TS samples a function from the GP's posterior and optimizes it. This balances exploration (due to posterior sampling) and exploitation (due to optimization of the sampled function). TS is particularly effective in non-stationary or high-dimensional landscapes.  
2. **Anisotropic Matérn kernel**: Ensures the GP adapts to the Rosenbrock valley's elongated structure by learning input-specific length scales.  
3. **Hybrid acquisition optimization**: Combines random candidates (Sobol) and local optimization (L-BFGS-B) to efficiently find extrema of sampled functions, avoiding the computational overhead of full batch methods.  
4. **Dynamic posterior sampling**: Each iteration samples a new function from the GP posterior, allowing the algorithm to adapt to the evolving model of the target function.  
The algorithm differs from the current population (e.g., DynamicLCBBO, MaternLCBBO) by using **probabilistic function sampling** instead of deterministic acquisition functions, which can better navigate the Rosenbrock valley's challenging curvature.

# Code  
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class ThompsonSamplingBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Initial Sobol points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n_points)
        # Scale to [-5, 5]
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)  # Avoid division by zero
        # Sample from posterior at X: N(mu, sigma^2)
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(50)
        acq_values = self._acquisition_function(candidates)
        # Select top 5 candidates for local optimization
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-5:]]  # Last 5 (highest acq values)

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()  # Maximize acq
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
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
        # Initial evaluation
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        # Fit initial model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            
            # Update best
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
            
            # Re-fit model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
```
## Feedback
 The algorithm ThompsonSamplingBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2943 with standard deviation 0.0170.

took 146.06 seconds to run.