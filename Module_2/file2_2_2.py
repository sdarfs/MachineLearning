import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.])
N = 500
lm = 0.02
batch_size = 50
Qe = 0.0  # начальное значение среднего эмпирического риска
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - batch_size)
    batch_x = coord_x[k:k + batch_size]
    batch_y = coord_y[k:k + batch_size]
    
    # Формируем матрицу признаков для мини-батча
    X_batch = np.column_stack([
        np.ones(batch_size),
        batch_x,
        batch_x ** 2,
        batch_x ** 3
    ])
    
    # Вычисляем предсказания и потери для мини-батча
    predictions = X_batch @ w
    losses = (predictions - batch_y) ** 2
    Qk = np.mean(losses)
    
    # Обновляем Qe
    Qe = lm * Qk + (1 - lm) * Qe
    
    # Вычисляем градиент для мини-батча
    gradient = (2 / batch_size) * X_batch.T @ (predictions - batch_y)
    
    # Обновляем веса с учетом индивидуального шага обучения для каждого параметра
    w -= eta * gradient

# Вычисляем итоговый средний эмпирический риск Q
X_all = np.column_stack([
    np.ones(sz),
    coord_x,
    coord_x ** 2,
    coord_x ** 3
])
predictions_all = X_all @ w
Q = np.mean((predictions_all - coord_y) ** 2)

# Сохраняем вектор параметров w в виде кортежа
w = tuple(w)
