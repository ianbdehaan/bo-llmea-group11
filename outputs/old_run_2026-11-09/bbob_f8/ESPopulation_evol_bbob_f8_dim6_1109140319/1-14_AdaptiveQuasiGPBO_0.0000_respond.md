**# Description**  
**AdaptiveQuasiGPBO** – a Bayesian‑Optimization framework that blends a space‑filling Sobol quasi‑Monte‑Carlo (QMC) initial design with a Gaussian‑Process (GP) surrogate (ARD Matern‑5/2 kernel) and an **Expected‑Improvement (EI)** acquisition.  
Key novelties over the previous Sobol‑Matern‑EI BO are:

* **Robust scalar handling** – all GP predictions and best‑so‑far values are stored as 1‑D arrays (`shape (n,)`) to avoid the “only 0‑dimensional arrays can be converted to Python scalars” error.  
* **Dynamic candidate pool** – the number of EI candidates grows with the remaining budget, guaranteeing enough diversity when the budget is large and keeping the cost low when the budget is tight.  
* **Adaptive batch size** – the algorithm automatically chooses a batch size of 1 or 2 (if at least 5 evaluations remain) to exploit cheap parallelism while staying within the evaluation budget.  
* **Selective GP re‑training** – the GP hyper‑parameters are re‑optimised only every ⌈dim/2⌉ iterations; in between a cheap “warm‑start” refit is performed, dramatically reducing CPU time.  
* **Improved local refinement** – each EI candidate is refined with L‑BFGS‑B using the **negative EI** gradient; the gradient is obtained analytically from the GP posterior for faster convergence.  
* **Kriging‑Believer batch selection** – after refining a point, its GP mean is temporarily added to the training set before picking the next point of the batch, preserving diversity without expensive inner optimisation.

The algorithm respects the strict evaluation budget, works for any dimension, and is fully vectorised with NumPy/Scikit‑Learn only.

---

**# Justification**  

| Component | Reasoning |
|-----------|-----------|
| **Sobol QMC initial design** | Provides a low‑discrepancy, space‑filling set of points that quickly explores the whole search box, crucial for the highly curved Rosenbrock valley. |
| **ARD Matern‑5/2 GP** | Automatic relevance determination (ARD) lets the GP adapt length‑scales per dimension, essential for anisotropic valleys; Matern‑5/2 balances smoothness and flexibility for Rosenbrock. |
| **Expected Improvement** | EI directly trades off exploitation (low mean) and exploration (high variance) and is cheap to evaluate on a large candidate set. |
| **Dynamic candidate pool** | The Rosenbrock function contains a narrow curved ridge; more candidates increase the chance of finding promising EI peaks when many evaluations remain, while a smaller pool saves time near the budget limit. |
| **Adaptive batch size** | A batch of 2 points can be evaluated in parallel on modern hardware and often yields a modest speed‑up; limiting to 2 keeps the Kriging‑Believer updates cheap and prevents premature convergence. |
| **Selective GP re‑training** | Full hyper‑parameter optimisation is the most expensive step; re‑optimising only every few iterations retains model quality while cutting runtime (the existing SobolEIHybridBO spends ~90 s). |
| **Analytic EI gradient** | Using the gradient for L‑BFGS‑B avoids numerical differentiation, leading to faster and more accurate local refinement of EI maxima. |
| **Kriging‑Believer** | Provides a cheap way to generate a diverse batch by pretending each newly selected point has its GP mean value, avoiding costly inner optimisation loops. |
| **Scalar handling fix** | Storing `y` as a 1‑D array (`shape (n,)`) eliminates the conversion error and simplifies indexing throughout the code. |

Together these improvements give a **computationally efficient** Bayesian optimiser that is expected to outperform the earlier Sobol‑Matern‑EI BO on the BBOB Function 8 (Rosenbrock, 6‑D) under a tight evaluation budget.

---

**# Code**  

```python
from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class AdaptiveQuasiGPBO:
    """
    AdaptiveQuasiGPBO
    -----------------
    * Sobol QMC initial design
    * ARD Matern‑5/2 GP surrogate
    * Expected Improvement (EI) acquisition
    * Dynamic candidate pool & adaptive batch size
    * Selective GP hyper‑parameter re‑training
    * Kriging‑Believer batch construction
    * Analytic EI gradient for L‑BFGS‑B refinement
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)          # total allowed function evaluations
        self.dim = int(dim)                # problem dimensionality
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])   # (2, dim)

        # History containers – y stored as 1‑D array for easy scalar conversion
        self.X: np.ndarray | None = None   # shape (n_samples, dim)
        self.y: np.ndarray | None = None   # shape (n_samples,)

        self.n_evals = 0                    # counter of used evaluations

        # Initial Sobol design size: at least 2*dim, never > budget/2
        self.n_init = min(max(2 * self.dim, 5), self.budget // 2)

        # GP re‑training schedule (every k iterations)
        self.retrain_interval = max(1, self.dim // 2)

    # ------------------------------------------------------------------ #
    # 1. Sobol sampling (scaled to bounds)
    # ------------------------------------------------------------------ #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Generate `n` Sobol points within the problem bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol produces points in [0, 1]^d; we need exactly n points
        m = int(np.ceil(np.log2(n)))               # power of two >= n
        raw = sampler.random_base2(m=m)[:n]
        lower, upper = self.bounds
        return lower + raw * (upper - lower)

    # ------------------------------------------------------------------ #
    # 2. GP surrogate (ARD Matern‑5/2)
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray,
                optimise_hyper: bool = True) -> GaussianProcessRegressor:
        """Fit (or refit) a GP. Hyper‑parameters are optimised only if requested."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                  Matern(length_scale=np.ones(self.dim),
                         length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                  WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2)))

        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0,
                                      normalize_y=True,
                                      n_restarts_optimizer=5 if optimise_hyper else 0,
                                      random_state=0)
        gp.fit(X, y)
        return gp

    # ------------------------------------------------------------------ #
    # 3. Expected Improvement (EI) and its gradient
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor,
            y_best: float) -> np.ndarray:
        """Vectorised EI for a set of candidates."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1)          # (n,)
        mu = mu.reshape(-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _ei_grad(self, x: np.ndarray, gp: GaussianProcessRegressor,
                 y_best: float) -> np.ndarray:
        """
        Analytic gradient of EI at a single point `x` (1‑D array, shape (dim,)).
        Uses GP posterior mean/variance derivatives from sklearn's GP.
        """
        # sklearn does not expose derivative directly → compute via
        # the kernel gradient (small overhead for dim <= 10)
        x = x.reshape(1, -1)               # (1, dim)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu[0]; sigma = sigma[0]

        if sigma <= 1e-12:
            return np.zeros(self.dim)

        # Derivatives of mean and variance
        K_trans = gp.kernel_(gp.X_train_, x)                # (n_train, 1)
        K_inv = gp.alpha_                                   # (n_train,) = (K+σ²I)^-1 y
        # d mu / d x = K_trans.T @ (K+σ²I)^-1 @ (X_train - x) * d_k
        # For Matern kernel the gradient can be obtained via sklearn's kernel gradient
        # using the private method _gradient_x (supported for Matern)
        dK_dx = gp.kernel_.gradient_x(gp.X_train_, x)       # (n_train, dim)

        # mean gradient
        dmu_dx = (K_trans.T @ (gp.alpha_[:, None] * dK_dx)).reshape(-1)

        # variance gradient: d sigma^2 / d x = 2 * (K_trans.T @ (K+σ²I)^-1 @ dK_dx)
        # compute v = (K+σ²I)^-1 K_trans
        v = gp._K_inv @ K_trans                               # (n_train, 1)
        dvar_dx = 2 * (v.T @ dK_dx).reshape(-1)

        sigma2 = sigma ** 2
        # EI gradient (see Jones et al., 1998)
        imp = y_best - mu
        Z = imp / sigma
        pdf = norm.pdf(Z)
        cdf = norm.cdf(Z)

        dei_dx = -dmu_dx * cdf + pdf * (dvar_dx / (2 * sigma))
        return dei_dx

    # ------------------------------------------------------------------ #
    # 4. Candidate generation (dynamic pool)
    # ------------------------------------------------------------------ #
    def _candidate_pool(self) -> np.ndarray:
        """Generate a candidate pool whose size adapts to remaining budget."""
        remaining = self.budget - self.n_evals
        # Heuristic: 5 * dim * log2(remaining+1) candidates, capped at 5000
        n_cand = min(int(5 * self.dim * np.log2(remaining + 1)), 5000)
        n_cand = max(n_cand, 200)               # never fewer than 200
        return self._sobol_samples(n_cand)

    # ------------------------------------------------------------------ #
    # 5. Local refinement of EI maxima (L‑BFGS‑B with analytic gradient)
    # ------------------------------------------------------------------ #
    def _refine_point(self, x0: np.ndarray, gp: GaussianProcessRegressor,
                      y_best: float) -> np.ndarray:
        """Run L‑BFGS‑B to maximise EI starting from `x0`."""
        bounds = [(low, high) for low, high in zip(self.bounds[0], self.bounds[1])]

        def neg_ei(x):
            return -self._ei(x.reshape(1, -1), gp, y_best)[0]

        def neg_ei_grad(x):
            return -self._ei_grad(x, gp, y_best)

        res = minimize(fun=neg_ei,
                       x0=x0,
                       jac=neg_ei_grad,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 40, 'ftol': 1e-6})
        return res.x if res.success else x0

    # ------------------------------------------------------------------ #
    # 6. Batch selection via Kriging‑Believer
    # ------------------------------------------------------------------ #
    def _select_batch(self, gp: GaussianProcessRegressor,
                      y_best: float, batch_sz: int) -> np.ndarray:
        """Return `batch_sz` new points using EI + Kriging‑Believer."""
        candidates = self._candidate_pool()
        ei_vals = self._ei(candidates, gp, y_best)

        # pick the best `batch_sz` candidates as start points
        start_idx = np.argpartition(-ei_vals, batch_sz)[:batch_sz]
        start_pts = candidates[start_idx]

        # refine each start point independently
        refined = [self._refine_point(pt, gp, y_best) for pt in start_pts]

        # Kriging‑Believer sequential addition
        batch = []
        X_tmp = self.X.copy()
        y_tmp = self.y.copy()
        gp_tmp = gp

        for pt in refined:
            batch.append(pt)
            # pretend observation = GP mean at pt
            mu_pt = gp_tmp.predict(pt.reshape(1, -1))[0]
            X_tmp = np.vstack([X_tmp, pt])
            y_tmp = np.append(y_tmp, mu_pt)
            # Re‑fit GP only for mean update (no hyper‑opt) – cheap
            gp_tmp = self._fit_gp(X_tmp, y_tmp, optimise_hyper=False)

        return np.array(batch)

    # ------------------------------------------------------------------ #
    # 7. Evaluation respecting the budget
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> np.ndarray:
        """Evaluate `func` on rows of X, truncating if budget would be exceeded."""
        n_allowed = min(X.shape[0], self.budget - self.n_evals)
        X = X[:n_allowed]
        y = np.empty(n_allowed, dtype=float)
        for i, xi in enumerate(X):
            y[i] = func(xi)
        self.n_evals += n_allowed
        return y

    # ------------------------------------------------------------------ #
    # 8. Main optimisation entry point
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run AdaptiveQuasiGPBO until the evaluation budget is exhausted.
        Returns (best_value, best_location).
        """

        # ---- Initial Sobol design -------------------------------------------------
        X_init = self._sobol_samples(self.n_init)
        y_init = self._evaluate(func, X_init)

        self.X = X_init.copy()
        self.y = y_init.copy()

        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ---- Main BO loop ---------------------------------------------------------
        iteration = 0
        while self.n_evals < self.budget:
            # Decide whether to re‑optimise GP hyper‑parameters
            optimise = (iteration % self.retrain_interval == 0)
            gp = self._fit_gp(self.X, self.y, optimise_hyper=optimise)

            # Adaptive batch size: 2 if at least 5 evaluations left, else 1
            remaining = self.budget - self.n_evals
            batch_sz = 2 if remaining >= 5 else 1

            # Select new points
            new_X = self._select_batch(gp, best_y, batch_sz)

            # Evaluate them (budget safety inside _evaluate)
            new_y = self._evaluate(func, new_X)

            # Update history
            self.X = np.vstack([self.X, new_X])
            self.y = np.append(self.y, new_y)

            # Update best-so‑far
            cur_best = np.argmin(new_y)
            if new_y[cur_best] < best_y:
                best_y = float(new_y[cur_best])
                best_x = new_X[cur_best].copy()

            iteration += 1

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveQuasiGPBO>", line 240, in __call__
 240->             new_X = self._select_batch(gp, best_y, batch_sz)
  File "<AdaptiveQuasiGPBO>", line 174, in _select_batch
 174->         refined = [self._refine_point(pt, gp, y_best) for pt in start_pts]
  File "<AdaptiveQuasiGPBO>", line 174, in <listcomp>
 174->         refined = [self._refine_point(pt, gp, y_best) for pt in start_pts]
  File "<AdaptiveQuasiGPBO>", line 152, in _refine_point
 152->         res = minimize(fun=neg_ei,
  File "<AdaptiveQuasiGPBO>", line 150, in neg_ei_grad
 150->             return -self._ei_grad(x, gp, y_best)
  File "<AdaptiveQuasiGPBO>", line 107, in _ei_grad
 105 |         # For Matern kernel the gradient can be obtained via sklearn's kernel gradient
 106 |         # using the private method _gradient_x (supported for Matern)
 107->         dK_dx = gp.kernel_.gradient_x(gp.X_train_, x)       # (n_train, dim)
 108 | 
 109 |         # mean gradient
AttributeError: 'Sum' object has no attribute 'gradient_x'
