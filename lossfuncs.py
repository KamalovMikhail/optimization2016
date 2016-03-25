__author__ = 'mikhail'

import numpy as np
import scipy.linalg as sp

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, -1, 1])
reg_coef = 0.5
w = np.array([0, 0])



def logistic(w, X, y, reg_coef, hess=False):
    yc = np.copy(y)
    Xk = np.copy(X)
    wc = np.copy(w)
    n = len(yc)

    fun_val = 1 / n * np.sum(np.log(1 + np.exp((y * X.T[:]).T[:] * w)), axis=0) + (reg_coef / 2) * np.power(sp.norm(w), 2)

    return fun_val

a = logistic(w, X, y, reg_coef)
print(a)