import numpy as np
x_test = [(5, -3), (-3, 8), (3, 6), (0, 0), (5, 3), (-3, -1), (-3, 3)]
w = np.array([-33, 9, 13])
a_sign = lambda x, w: -1 if x[0]*w[0] + x[1]*w[1] + x[2]*w[2] < 0 else 1
x_test_new = np.array([[1, x1, x2] for x1, x2 in x_test])
predict = [a_sign(x, w) for x in x_test_new]
