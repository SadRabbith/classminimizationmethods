#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import rosen
import random

x = np.arange(-2, 2, 0.1)
y = np.arange(-1, 3, 0.1)
X, Y = np.meshgrid(x, y)

z = rosen((X,Y)) # this is our function
plt.pcolormesh(X, Y, z, norm='log', vmin=1e-3)
c = plt.colorbar()
plt.show()
#%%
def nelder_mead(func, max_iter=500):
    """ 
    func: rosen function
    """
    alpha=1.0
    gamma=2.0
    rho=0.5
    sigma=0.5

    # Build the triangle
    pos1 = []
    triangle = []
    x1 = random.uniform(-2.0, 2.0)
    pos1.append(x1)
    y1 = random.uniform(-2.0, 2.0)
    pos1.append(y1)
    triangle.append(pos1)
    pos2 = []
    x2 = random.uniform(-2.0, 2.0)
    pos2.append(x2)
    y2 = random.uniform(-2.0, 2.0)
    pos2.append(y2)
    triangle.append(pos2)
    pos3 = []
    x3 = random.uniform(-2.0, 2.0)
    pos3.append(x3)
    y3 = random.uniform(-2.0, 2.0)
    pos3.append(y3)
    triangle.append(pos3)
    triangle = np.array(triangle)

    ##print(triangle)

    for iteration in range(max_iter):
        # find the best and worst points
        vals = np.array([func(x) for x in triangle])
        idx = np.argsort(vals)
        triangle = triangle[idx]
        vals = vals[idx]

        # calculate centroid excluding worst point
        centroid = np.mean(triangle[:-1], axis=0)

        # reflect the worst point
        xr = centroid + alpha * (centroid - triangle[-1])
        fr = func(xr)
        # decide whether to expand or contract
        if vals[0] <= fr < vals[-2]:
            triangle[-1] = xr
        elif fr < vals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            triangle[-1] = xe if fe < fr else xr
        else:
            # Contraction
            xc = centroid + rho * (triangle[-1] - centroid)
            fc = func(xc)
            if fc < vals[-1]:
                triangle[-1] = xc
            else:
                # Shrink
                best = triangle[0]
                triangle = best + sigma * (triangle - best)

    return triangle[0], func(triangle[0]), iteration

# impliment on rosen func
x_start = np.array([-2, 2.0])
xmin, fmin, iters = nelder_mead(rosen)

print(f"Minimum found at {xmin} after {iters} iterations")

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosen((X, Y))

plt.show()
plt.figure(figsize=(7,5))
plt.pcolormesh(X, Y, Z, norm='log', vmin=1e-3)
plt.colorbar()
plt.plot(xmin[0], xmin[1], 'ro', label='Minimum found')
plt.legend()
plt.title("Nelder–Mead Minimization of the Rosenbrock Function")
plt.show()

# %%
