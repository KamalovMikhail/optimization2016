__author__ = 'mikhail'

import numpy as np
from numpy.linalg import norm


A = np.array([[1, 0], [0, 2]])
b = np.array([1, 6])
x0 = np.array([0, 0])

matvec = (lambda x: A.dot(x))

def cg(matvec, b, x0, tol=1e-5, max_iter=None, disp=False, trace=False):
    xk = np.copy(x0)
    bk = np.copy(b)
    r0 = matvec(xk) - bk
    d0 = - r0
    k = 0
    status = 1
    for i in range(0, len(xk)):
        alpha_k = np.dot(r0.T, r0) / np.dot(d0.T, matvec(d0))
        xk = xk + alpha_k * d0
        rk = r0 + alpha_k * matvec(d0)
        betta_k = rk.T.dot(rk) / r0.T.dot(r0)
        d0 = - rk + betta_k * d0
        k = k + 1
        r0 = rk
        if norm(matvec(xk) - b) <= tol:
            status = 0
            break
    return xk, status


a = cg(matvec, b, x0)
print(a)




