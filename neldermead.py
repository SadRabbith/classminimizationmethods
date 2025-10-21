import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import rosen

x = np.arange(-2, 2, 0.1)
y = np.arange(-1, 3, 0.1)
X, Y = np.meshgrid(x, y)

z = rosen((X,Y)) # this is our function
plt.pcolormesh(X, Y, z, norm='log', vmin=1e-3)
c = plt.colorbar()
plt.show()

## first we reflect
def nelder_mead(func):
    # idk what this is
    alpha = 4

    # test points in range of rosen
    test_points = np.random.rand(-2, 3, size = 3)
    ### NOTE: im just calling in the rosen fucntion so right now this is just running to find the smallest number in the given range
    # defining the max and min
    x1 = min(test_points)
    test_points.remove(x1)
    xn = max(test_points)
    test_points.remove(xn)

    # find centroid of all points accept maximum
    x0 = test_points(0) + x1 / 2
    
    # finding reflection point
    xr = x0 + alpha(x0 - xn)

    # checking that its better than the second worst
    if xr < test_points(0):
        # this means its better than the best
        if xr < x1:
            # here we want to just redo what we just did 
        # this means its somewhere in the middle
        else: 

### contract

### expand