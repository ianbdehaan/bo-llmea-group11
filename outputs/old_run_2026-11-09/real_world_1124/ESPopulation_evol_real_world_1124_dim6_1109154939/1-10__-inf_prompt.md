You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of Real World Problem 1124 (Frequency Modulated Sound Waves, 6D). Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- QMC_EI_BO: 0.0000, 0.00 seconds, **  
A **Quasi‑Monte‑Carlo‑initialized, batch Expected‑Improvement Bayesian Optimiser (QMC‑EI‑BO)**  

- **Quasi‑Monte‑Carlo (Sobol) initial design** – space‑filling, deterministic sampling that quickly yields a diverse set of points without the randomness of pure Monte‑Carlo.  
- **Gaussian Process surrogate** (scikit‑learn) with a Matern‑5/2 kernel – flexible, cheap to train, and provides analytical mean/variance needed for acquisition.  
- **Expected Improvement (EI)** as acquisition – balances exploitation (low mean) and exploration (high variance).  
- **Batch selection via candidate‑screening** – a large Sobol candidate set is generated, EI is evaluated on all candidates, and the top *k* points are taken.  This “cheap‑screen‑then‑pick” strategy avoids costly inner optimisation of the acquisition and is well‑suited for modest budgets (e.g. BBOB 6‑D).  
- **Kriging‑Believer update** for the batch – after picking each point the GP posterior is temporarily updated with the point’s predicted mean, reducing the chance of picking duplicate locations.  

The algorithm respects the strict evaluation budget, works in any dimension, and needs only NumPy, SciPy and scikit‑learn – all allowed libraries.

---

**


The selected solution to update is:

With code:
```python

```
No code was extracted. The code should be encapsulated with ``` in your response.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

