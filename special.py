__author__ = 'mikhail'

import numpy as np


def grad_finite_diff(func, x, eps=1e-8):

    e = np.zeros(len(x))
    gradient = np.array([])
    for i in range(len(x)):
        e[i] = 1
        gradient = np.append(gradient, ((func(x + eps * e) - func(x)) / eps))
        e[i] = 0
    return gradient


def hess_finite_diff(func, x, eps=1e-5):
    e1, e2, hessian = np.zeros(len(x)), np.zeros(len(x)), np.empty((0, len(x)))

    for i in range(len(x)):
        row = np.array([])
        for j in range(len(x)):
            e2[j], e1[i] = 1, 1
            row = np.append(row, ((func(x + eps * e1 + eps * e2) - func(x + eps * e1) - func(x + eps * e2) + func(x)) / (eps * eps)))
            e1[i], e2[j] = 0, 0
        hessian = np.append(hessian, np.array([row]), axis=0)

    return hessian


