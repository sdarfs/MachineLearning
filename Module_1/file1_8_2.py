import numpy as np


def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

model_a = lambda x, w: w @ np.array([1, x, x**2, np.cos(2*x), np.sin(2*x)])
loss = lambda a, y: abs(a-y)

coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x) 

w = np.array([-1.59, -0.69, 0.278, 0.497, -0.106])

Q = 1/sz * sum([loss(model_a(x, w), func(x)) for x in coord_x])
