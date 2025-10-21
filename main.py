import numpy as np 
import matplotlib.pyplot as plt 
import statistics 

#brute force method 
#gradient descent method 
#newton's method 

#2D rosenbrock => f(x,y) = (1-x)**2 +100(y-x)**2


def all_solutions(x_values, y_values):
    rosen = []
    for x_value in x_values:
        for y_value in y_values: 
            fun = (1 -x_value)**2 + 100(y_value - x_value)**2 
            rosen.append(fun) 
    
    max = 0

    for values in range(len(rosen)): 
        if max > rosen[values]: 
            values = values+ 1 
        elif max < rosen[values]: 
            max = rosen[values]
            values = values + 1 
    print(max)




