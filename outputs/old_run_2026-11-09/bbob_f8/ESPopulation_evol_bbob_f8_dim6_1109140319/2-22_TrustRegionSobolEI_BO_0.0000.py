bo = TrustRegionSobolEI_BO(budget=200, dim=6)
best_val, best_x = bo(my_rosenbrock)   # `my_rosenbrock` takes a 1‑D np.ndarray and returns a float
