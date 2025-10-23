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

def gradient_descent_multivariate(x_init, y_init, learning_rate, num_iterations):

    params = np.array([x_init, y_init], dtype=float) # Current (x, y) point
    
    
    x_history = [x_init]
    y_history = [y_init]
    
    
    # history = [params.copy()] 

    for i in range(num_iterations):
        
        
        x_current, y_current = params[0], params[1] 
        
        
        gradient = np.array(df(x_current, y_current), dtype=float)
        
        
        params -= learning_rate * gradient
        
       
        x_history.append(params[0])
        y_history.append(params[1])

    # Return the final parameters and the two history lists
    return params, x_history, y_history

l_rate = 0.001
num = 100000
#lower left
final_params_ll, x_history_ll, y_history_ll = gradient_descent_multivariate(-2.0, -1.0, l_rate , num )

#lower right
final_params_lr, x_history_lr, y_history_lr = gradient_descent_multivariate(2.0, -1.0, l_rate , num)

#upper left
final_params_ul, x_history_ul, y_history_ul = gradient_descent_multivariate(-2.0, 3.0, l_rate , num)

#upper right
final_params_ur, x_history_ur, y_history_ur = gradient_descent_multivariate(2.0, 3.0, l_rate , num)

print(f'the final parameters = {final_params_ur}')
print(f"history = {x_history_ur, y_history_ur}")

plt.figure()

plt.plot(x_history_ur, y_history_ur, color = 'blue', label = 'start = (2,3)')
plt.plot(x_history_ul, y_history_ul, color = 'red', label = 'start = (-2, 3)')
plt.plot(x_history_ll, y_history_ll, color = 'green', label = 'start = (-2, -1)')
plt.plot(x_history_lr, y_history_lr, color = 'yellow', label = 'start = (2, -1)')

plt.plot(x_history_ur[-1], y_history_ur[-1], 'kx', markersize=12, markeredgewidth=2, color = 'deeppink')
plt.plot(x_history_ul[-1], y_history_ul[-1], 'kx', markersize=12, markeredgewidth=2, color = 'orange')
plt.plot(x_history_ll[-1], y_history_ll[-1], 'kx', markersize=12, markeredgewidth=2, color = 'purple')
plt.plot(x_history_lr[-1], y_history_lr[-1], 'kx', markersize=12, markeredgewidth=2, color = 'gray')

plt.title(f"Gradient Descent, learning rate = {l_rate} and number of iterations = {num}")
plt.legend(loc='upper right')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

