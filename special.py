__author__ = 'mikhail'

from lossfuncs import logistic

def grad_finite_diff(func, x, eps=1e-8):
    func(x + eps * (0 )) - logistic()