import numpy as np 
import matplotlib.pyplot as plt 
import statistics 

#2D rosenbrock => f(x,y) = (1-x)**2 +100(y-x**2)**2

def f(x,y):
    return (1 - x)**2 + 100 * (y - x**2)**2

#derivative 

def df(x,y): 
    df_dx = 2 * (x - 1) + 400 * x * (x**2 - y)
    df_dy = 200 * (y - x**2)

    return df_dx , df_dy

def gradient_descent_multivariate(x_init, y_init ,  learning_rate , num_iterations):
    """
    Performs gradient descent to minimize a multivariate function.
    """
    params = np.array(x_init, y_init, dtype=float) # Ensure float type
    history = [params.copy()] # Store history of parameters

    for i in range(num_iterations):
        gradient = df(x_init, y_init)
        params -= learning_rate * gradient
        history.append(params.copy())

    return params, history
