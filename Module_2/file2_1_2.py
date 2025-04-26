# Необходимо выполнить аппроксимацию (восстановление) на интервале [-5, 5]

import numpy as np


def func(x):
    """Исходная функция"""
    return 0.1 * x**2 - np.sin(x) + 5.


def model(x, w):
    """Модель a(x)"""
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3


def loss(y_true, y_pred):
    """Функция потерь (среднеквадратичная ошибка)"""
    return np.mean((y_true - y_pred) ** 2)


def gradients(x, y_true, w):
    """Градиенты для параметров w"""
    y_pred = model(x, w)
    error = y_pred - y_true
    grad_w0 = 2 * np.mean(error)
    grad_w1 = 2 * np.mean(error * x)
    grad_w2 = 2 * np.mean(error * x**2)
    grad_w3 = 2 * np.mean(error * x**3)
    return np.array([grad_w0, grad_w1, grad_w2, grad_w3])


coord_x = np.arange(-5.0, 5.0, 0.1)  # Значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x)  # Значения функции по оси ординат

sz = len(coord_x)  # Количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # Шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # Начальные значения параметров модели
N = 200  # Число итераций градиентного алгоритма

# Градиентный спуск
for i in range(N):
    grads = gradients(coord_x, coord_y, w)
    w -= eta * grads  # Обновление параметров

Q = loss(model(coord_x, w), coord_y)
