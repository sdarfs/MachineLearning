# Необходимо выполнить аппроксимацию (восстановление) на интервале [-5, 5] с помощью алгоритма стохастического градиентного спуска SGD 
import numpy as np

def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)
sz = len(coord_x)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02
Qe = 0.0  # начальное значение среднего эмпирического риска
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz)
    x = coord_x[k]
    y = coord_y[k]
    
    # Формируем вектор признаков
    x_features = np.array([1, x, x**2, np.cos(2*x), np.sin(2*x)])
    
    # Вычисляем предсказание и потерю
    prediction = np.dot(w, x_features)
    loss = (prediction - y)**2
    
    # Обновляем Qe
    Qe = lm * loss + (1 - lm) * Qe
    
    # Вычисляем градиент
    gradient = 2 * (prediction - y) * x_features
    
    # Обновляем веса с учетом индивидуального шага обучения для каждого параметра
    w -= eta * gradient

# Вычисляем итоговый средний эмпирический риск Q
total_loss = 0.0
for i in range(sz):
    x = coord_x[i]
    y = coord_y[i]
    x_features = np.array([1, x, x**2, np.cos(2*x), np.sin(2*x)])
    prediction = np.dot(w, x_features)
    total_loss += (prediction - y)**2

Q = total_loss / sz

# Сохраняем вектор параметров w в виде списка или кортежа
w = tuple(w)
