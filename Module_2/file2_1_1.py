# Напишите программу поиска точки минимума функции с помощью градиентного алгоритма
import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3

def df(x):
    return 0.5 + 0.4 * x - 0.3 * x ** 2
  
x0 = -4
lmb = 0.01
N = 200

x = x0
for _ in range(N):
    x -= lmb * df(x)
