import numpy as np

X_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
y_test = np.array([1, 1, 1, -1, -1, -1, -1])
w = np.array([-8/3, -2/3, 1])

margin = y_test * (np.dot(X_test, w[1:]) + w[0])

# Подсчет количества неправильных классификаций (где margin < 0)
Q = np.sum(margin < 0) #  показатель качества  в соответствии с формулой: Q = sum([M_i < 0])
print(Q)
