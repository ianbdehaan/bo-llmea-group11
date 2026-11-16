**# Description**  
A **Quasi‑Monte‑Carlo‑initialized, batch Expected‑Improvement Bayesian Optimiser (QMC‑EI‑BO)**  

- **Quasi‑Monte‑Carlo (Sobol) initial design** – space‑filling, deterministic sampling that quickly yields a diverse set of points without the randomness of pure Monte‑Carlo.  
- **Gaussian Process surrogate** (scikit‑learn) with a Matern‑5/2 kernel – flexible, cheap to train, and provides analytical mean/variance needed for acquisition.  
- **Expected Improvement (EI)** as acquisition – balances exploitation (low mean) and exploration (high variance).  
- **Batch selection via candidate‑screening** – a large Sobol candidate set is generated, EI is evaluated on all candidates, and the top *k* points are taken.  This “cheap‑screen‑then‑pick” strategy avoids costly inner optimisation of the acquisition and is well‑suited for modest budgets (e.g. BBOB 6‑D).  
- **Kriging‑Believer update** for the batch – after picking each point the GP posterior is temporarily updated with the point’s predicted mean, reducing the chance of picking duplicate locations.  

The algorithm respects the strict evaluation budget, works in any dimension, and needs only NumPy, SciPy and scikit‑learn – all allowed libraries.

---

**# Justification**  

1. **Sobol (QMC) sampling** gives low‑discrepancy points that cover the bounded domain more uniformly than random Latin‑Hypercube designs, leading to a stronger initial model with fewer evaluations – crucial when the total budget is limited (typical BBOB budgets are a few hundred evaluations).  

2. **Matern‑5/2 kernel** captures moderately smooth objective functions (the FM‑Sound‑Waves benchmark exhibits smooth but non‑trivial curvature) while keeping the hyper‑parameter space small, allowing fast optimisation of the GP hyper‑parameters.  

3. **Expected Improvement** is analytically tractable with a GP and directly targets the global minimum, making it the standard acquisition for black‑box minimisation.  

4. **Batch acquisition via screening** avoids the expensive inner optimisation loops often used in BO (e.g. gradient‑based optimisation of EI). By generating a dense Sobol candidate set (e.g. 5 000 points) and selecting the best *k*, we obtain a good batch with negligible extra cost.  

5. **Kriging‑Believer** (sequentially “pretend” that each newly selected point will return its GP mean) decorrelates the batch points, improving diversity without a full multi‑point optimisation of EI.  

6. **Computational efficiency** – all heavy work is vectorised (candidate generation, EI evaluation). The GP is refitted only once per BO iteration (after the whole batch), keeping the overhead small compared with the evaluation budget.  

Overall, the method is **novel in its combination of Sobol‑based initialisation, cheap candidate‑screening batch EI, and Kriging‑Believer updating**, delivering a fast and reliable optimiser for the 6‑D FM‑Sound‑Waves BBOB problem.

---

**# Code**  

```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class QMC_EI_BO:
    """
    Quasi‑Monte‑Carlo initialised Bayesian Optimiser with batch Expected Improvement.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total number of allowed f‑evaluations
        self.dim = dim                      # problem dimensionality
        # bounds shape (2, dim): lower row, upper row
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None           # evaluated points, shape (n, dim)
        self.y: np.ndarray = None           # corresponding objective values, shape (n, 1)
        self.n_evals = 0                    # counter of performed evaluations

        # initial design size – at least 2*dim, but not larger than budget/2
        self.n_init = min(max(2 * dim, 10), self.budget // 2)

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Generate `n_points` quasi‑Monte‑Carlo Sobol samples inside the bounds.
        Returns array of shape (n_points, dim).
        """
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0,1]^d; we scale to bounds
        unit_samples = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
        unit_samples = unit_samples[:n_points]               # keep exactly n_points
        scaled = qmc.scale(unit_samples,
                           self.bounds[0],
                           self.bounds[1])
        return scaled

    # ------------------------------------------------------------------ #
    # Surrogate model
    # ------------------------------------------------------------------ #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a Gaussian Process with a Matern‑5/2 kernel.
        Returns the trained GP model.
        """
        # Kernel: constant * Matern + WhiteNoise (for numerical stability)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-1))

        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Acquisition – Expected Improvement
    # ------------------------------------------------------------------ #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor,
                               y_best: float) -> np.ndarray:
        """
        Compute Expected Improvement at points X.
        Returns array of shape (n_points, 1).
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # Avoid division by zero
        with np.errstate(divide='ignore'):
            imp = y_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    # ------------------------------------------------------------------ #
    # Batch selection
    # ------------------------------------------------------------------ #
    def _select_next_points(self, gp: GaussianProcessRegressor,
                            batch_size: int) -> np.ndarray:
        """
        Generate a large Sobol candidate set, evaluate EI, and pick the top `batch_size`
        points using a Kriging‑Believer scheme.
        Returns array of shape (batch_size, dim).
        """
        # Number of candidates – a few thousand gives a smooth EI landscape
        n_candidates = max(5000, 10 * self.dim)

        candidates = self._sample_points(n_candidates)

        # Current best observed value
        y_best = np.min(self.y)

        selected = []
        temp_X = self.X.copy()
        temp_y = self.y.copy()

        for _ in range(batch_size):
            ei = self._acquisition_function(candidates, gp, y_best)
            # Mask already selected points
            if selected:
                mask = np.any(np.isclose(candidates, selected).all(axis=1), axis=0)
                ei[mask] = -np.inf

            idx = np.argmax(ei)
            x_next = candidates[idx:idx + 1]          # shape (1, dim)
            selected.append(x_next.squeeze())
            # Kriging‑Believer: pretend the observation will be the GP mean
            mu = gp.predict(x_next)
            temp_X = np.vstack([temp_X, x_next])
            temp_y = np.vstack([temp_y, mu.reshape(-1, 1)])
            # Re‑fit a *temporary* GP for the next selection (cheap because data size is tiny)
            gp = self._fit_model(temp_X, temp_y)

        return np.array(selected)

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """
        Evaluate the black‑box function on each row of X.
        Returns an array of shape (n_points, 1).
        """
        y_vals = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += len(X)
        return y_vals

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Append newly evaluated points to the internal data containers.
        """
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # ------------------------------------------------------------------ #
    # Main optimisation loop
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """
        Run the Bayesian optimisation loop.
        Returns (best_y, best_x).
        """
        # -------------------- Initial design -------------------- #
        init_n = min(self.n_init, self.budget)   # safety against tiny budgets
        X_init = self._sample_points(init_n)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Fit the first surrogate
        gp = self._fit_model(self.X, self.y)

        # -------------------- BO iterations -------------------- #
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals
            # Choose a batch size (max 5, but not larger than remaining)
            batch_size = min(5, remaining)

            # Propose next points
            X_next = self._select_next_points(gp, batch_size)

            # Evaluate them (respect budget)
            y_next = self._evaluate_points(func, X_next)

            # Update data and re‑fit model
            self._update_eval_points(X_next, y_next)
            gp = self._fit_model(self.X, self.y)

        # -------------------- Return best found -------------------- #
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<QMC_EI_BO>", line 173, in __call__
 173->             X_next = self._select_next_points(gp, batch_size)
  File "<QMC_EI_BO>", line 110, in _select_next_points
 108 |             # Mask already selected points
 109 |             if selected:
 110->                 mask = np.any(np.isclose(candidates, selected).all(axis=1), axis=0)
 111 |                 ei[mask] = -np.inf
 112 | 
ValueError: operands could not be broadcast together with shapes (5000,6) (2,6) 
