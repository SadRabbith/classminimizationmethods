import numpy as np 
import matplotlib.pyplot as plt 
import statistics 

#brute force method 
#gradient descent method 
#newton's method 

#2D rosenbrock => f(x,y) = (1-x)**2 +100(y-x)**2

x_min, x_max = -2.0, 2.0
y_min, y_max = -1.0, 3.0


N_DIVISIONS = 50 

x_values = np.linspace(x_min, x_max, N_DIVISIONS)
y_values = np.linspace(y_min, y_max, N_DIVISIONS)

def brute_force(x_values, y_values):
    rosen = dict()
    for x_value in x_values:
        for y_value in y_values: 
            fun = (1 -x_value)**2 + 100*(y_value - x_value**2)**2
            rosen[(x_value, y_value)] = fun
    
    min_value = float('inf')
    min_point = None

    for key, value in rosen.items():
        if value < min_value:
            min_value = value
            min_point = key 

    print(f"Minimum value: {min_value} at (x, y) = {min_point}")
    return rosen

brute_force(x_values=x_values, y_values=y_values) 
