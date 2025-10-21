import numpy as np 
import matplotlib.pyplot as plt 
import statistics 

#brute force method 
#gradient descent method 
#newton's method 

#2D rosenbrock => f(x,y) = (1-x)**2 +100(y-x)**2

x_values = [-2, -1 , 0, 1, 2]
y_values = [-1, 0, 1 , 2, 3]


def all_solutions(x_values, y_values):
    rosen = dict()
    for x_value in x_values:
        for y_value in y_values: 
            fun = (1 -x_value)**2 + 100*(y_value - x_value**2)**2
            rosen[(x_value, y_value)] = fun
    
    max_value = float('-inf')
    max_point = None

    for key, value in rosen.items():
        if value > max_value:
            max_value = value
            max_point = key

    print(f"Maximum value: {max_value} at (x, y) = {max_point}")
    return rosen

all_solutions(x_values=x_values, y_values=y_values) 
