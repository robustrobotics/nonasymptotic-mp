from nonasymptotic.util import compute_numerical_bound
import numpy as np
import matplotlib.pyplot as plt

bounds = []
space = list(np.linspace(0.179*0.2, 0.179*2, 50))
for delta in space:
    n, _ = compute_numerical_bound(delta, 0.9, 4, 2, None)
    bounds.append(n)
plt.plot(space, bounds)
plt.show()