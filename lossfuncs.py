__author__ = 'mikhail'

import numpy as np
import scipy.linalg as sp
import scipy.special as spes
import scipy.sparse as sps
from special import grad_finite_diff, hess_finite_diff

#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([1, 1, -1, 1])
#reg_coef = 0.5
#w = np.array([0, 0])






def logistic(w, X, y, reg_coef, hess=False):
    yc, Xk, wc, n = np.copy(y), np.copy(X), np.copy(w), len(y)
    c = -yc * Xk.dot(wc)

    # function
    fun_val = 1 / n * np.sum(np.logaddexp(np.zeros(len(c)), c)) + (reg_coef / 2) * np.power(sp.norm(wc), 2)

    # gradient
    sigma = spes.expit(c)
    grad = 1 / n * np.sum((-yc * Xk.T * sigma).T, axis=0) + reg_coef * wc

    # hessian
    if hess:
        d = np.multiply(sigma, (np.ones(n) - sigma))
        D = sps.diags(d, offsets=0)
        hessian = (1 / n) * np.dot(Xk.T, D.dot(Xk)) + reg_coef * np.diag(np.ones(len(wc)), 0)

    if hess:
        return fun_val, grad, hessian
    else:
        return fun_val, grad


X = np.random.random((16, 4))
wc = np.array([0.523, 0.444, -1, 0.222])
Y = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1])

f, g, h = logistic(w=wc, X=X, y=Y, reg_coef=0.5, hess=True)

print(g)
print(h)

func = (lambda x: logistic(w=x, X=X, y=Y, reg_coef=0.5)[0])
print(grad_finite_diff(func, wc))
print(hess_finite_diff(func, wc))



#a,b,h = logistic(w, X, y, reg_coef, hess=True)

#print(a)
# key phrase extraction
#print("!!")
#print(a, b,h)

#print(h)