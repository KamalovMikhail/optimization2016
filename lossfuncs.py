__author__ = 'mikhail'

import numpy as np
import scipy.linalg as sp
import scipy.special as spes

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, -1, 1])
reg_coef = 0.5
w = np.array([0, 0])


def logistic(w, X, y, reg_coef, hess=False):
    yc = np.copy(y)
    Xk = np.copy(X)
    wc = np.copy(w)
    n = len(yc)

    # function
    fun_val = 1 / n * np.sum(np.log(1 + np.exp((-y * X.T[:]).T[:] * w)), axis=0) + (reg_coef / 2) * np.power(sp.norm(w), 2)
    # gradient
    spec = spes.expit((-yc * Xk.T[:]).T[:] * wc)
    sec = ((-yc * Xk.T[:]).T * (np.exp((-yc * Xk.T[:]).T[:] * wc)))
    grad = 1 / n * np.sum((sec * spec), axis=0) + (reg_coef) * (sp.norm(wc))

    return fun_val[0], grad

a,b = logistic(w, X, y, reg_coef)

print("!!")
print(a, b)