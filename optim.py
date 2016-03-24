__author__ = 'mikhail'


import numpy as np


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

    for i in range(0, len(xk)):
        print(i)
        numerator_a = r0.T.dot(r0)
        denominator_a = d0.T.dot(matvec(d0))
        alpha_k = np.ndarray(np.divide(numerator_a, denominator_a))

        xk = xk + np.dot(alpha_k, d0)
        rk = r0 + np.dot(alpha_k, matvec(d0))
        numerator_b = rk.T.dot(rk)
        denominator_b = r0.T.dot(r0)
        betta_k = np.divide(numerator_b, denominator_b)

        d0 = - rk + betta_k.dot(d0)
        k = k + 1

        r0 = rk

        if np.norm(matvec(xk) - b) <= tol:
            break
    return xk


a = cg(matvec, b, x0)




