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