import numpy as np


def func(x):
    return 0.1 * x**2 - np.sin(x) + 0.1 * np.cos(x * 5) + 1.


coord_x = np.arange(-5.0, 5.0, 0.1)  # значения отсчетов по оси абсцисс
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # общее число отсчетов

# Параметры модели
w = np.array([1.11, -0.26, 0.061, 0.0226, 0.00178])

# Вычисление значений модели a(x, w)
a_values = (
    w[0] + 
    w[1] * coord_x + 
    w[2] * coord_x**2 + 
    w[3] * coord_x**3 + 
    w[4] * coord_x**4
)

# Вычисление квадратичных потерь
losses = (a_values - coord_y) ** 2

# Вычисление среднего эмпирического риска Q
Q = np.mean(losses)
