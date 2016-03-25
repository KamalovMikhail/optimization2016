__author__ = 'mikhail'

import scipy.linalg as sp
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

## b = np.array([1, 6])
## x0 = np.array([0, 0])



def cg(matvec, b, x0, tol=1e-5, max_iter=None, disp=False, trace=False):
    xk = np.copy(x0)
    bk = np.copy(b)
    r0 = matvec(xk) - bk
    d0 = - r0
    k = 1
    nev = norm(r0)
    status = 1
    n = len(xk) if (max_iter == None) else max_iter

    if trace:
        hist = {'it': np.array([]), 'n_evals': np.array([])}
        hist['it'] = np.append(hist['it'], k)
        hist['n_evals'] = np.append(hist['n_evals'], nev)

    if disp:
        print("%10s %15s" % ('iter', 'res_rat'))
        print("%10d %15e" % (k, nev))

    for i in range(0, n):
        alpha_k = np.dot(r0.T, r0) / np.dot(d0.T, matvec(d0))
        xk = xk + alpha_k * d0
        rk = r0 + alpha_k * matvec(d0)
        betta_k = rk.T.dot(rk) / r0.T.dot(r0)
        d0 = - rk + betta_k * d0
        k = k + 1
        r0 = rk
        nev = norm(matvec(xk) - b)
        if disp:
            print("%10d %15e" % (k, nev))

        if trace:
            hist['it'] = np.append(hist['it'], k)
            hist['n_evals'] = np.append(hist['n_evals'], nev)

        if nev <= tol:
            status = 0
            break

    if trace:
        return xk, status, hist
    else:
        return xk, status



def draw_plot(k, n, s):
    color = ['b', 'r', 'g']
    matvec = (lambda x: A.dot(x))

    c = 0
    for i in k:
        for sch in range(1, s):
            B = np.random.randn(n)
            X0 = np.zeros(n)
            V = sp.orth(np.random.randn(n, n))
            AL = np.zeros((n, n), float)
            np.fill_diagonal(AL, np.random.uniform(1, i, n), wrap=True)
            A = V.dot(AL).dot(V.T)
            x, r, l = cg(matvec, B, X0, disp=True, trace=True)

            plt.semilogy(l['n_evals'], l['it'], color[c], label = 'CG' ,alpha = 0.8)
        c = c + 1
    plt.show()


draw_plot([4, 14, 20], 10, 4)



