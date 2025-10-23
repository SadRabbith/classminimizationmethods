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

def gradient_descent_multivariate_corrected(x_init, y_init, learning_rate, num_iterations):

    params = np.array([x_init, y_init], dtype=float) # Ensure float type
    history = [params.copy()] 

    for i in range(num_iterations):
        
        x_current, y_current = params[0], params[1] 
        
       
        gradient = np.array(df(x_current, y_current), dtype=float)
        
      
        params -= learning_rate * gradient
        
       
        history.append(params.copy())

    #print(history[num_iterations])
    return params, history


final_params, history = gradient_descent_multivariate_corrected(-2.0, -1.0, 0.001, 10000)

print(f'the final parameters = {final_params}')
print(f"history = {history}")