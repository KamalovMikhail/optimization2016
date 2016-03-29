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
    fun_val = 1 / n * np.sum(np.log(1 + np.exp((-y * X.T).T[:] * w)), axis=0) + (reg_coef / 2) * np.power(sp.norm(w), 2)

    # gradient
    sigma = spes.expit(-y * X.dot(w))
    grad = 1 / n * np.sum((-y * X.T * sigma).T, axis=0) + reg_coef * wc

    # hessian
    if hess:
        d = sigma * (1 - sigma)
        D = np.diag(d)
        hessian = (1 / n) * np.dot(Xk.T, D).dot(X) + np.diag((reg_coef + np.zeros(len(wc)) ))

    if hess:
        return fun_val[0], grad, hessian
    else:
        return fun_val[0], grad

a,b,h = logistic(w, X, y, reg_coef, hess=True)

#print("!!")
print(a, b,h)

print(h)