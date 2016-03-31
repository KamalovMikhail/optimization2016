__author__ = 'mikhail'

from sklearn.datasets import load_svmlight_file
from optim import gd, ncg, newton
from lossfuncs import logistic
import numpy as np
import matplotlib.pyplot as plt

w = np.zeros(5000)


Mat = load_svmlight_file("/home/mikhail/Documents/optimtzation2016/gisette_scale")
X = Mat[0].toarray()
Y = Mat[1]


func = (lambda x: logistic(x, X, Y, 1 / 6000))

func_hess = (lambda x: logistic(x, X, Y, 1 / 6000, hess=True))

color = ['b', 'r', 'g']
xg, fg, sg, hg = gd(func, w, trace=True)
xn, fn, sn, hn = newton(func_hess, w, trace=True)
xnc, fnc, snc, hnc = ncg(func, w, trace=True)

plt.semilogy(hg['n_evals'], hg['f'] - fg, color=color[0], label='GD')
plt.semilogy(hn['n_evals'], hn['f'] - fn, color=color[1], label='Newton')
plt.semilogy(hnc['n_evals'], hnc['f'] - fnc, color=color[2], label='NCG')
plt.xlabel("oracul coals")
plt.ylabel("function values")
plt.title("Function value (oracul)")
plt.grid()
plt.legend(loc='best')
plt.show()

plt.semilogy(hg['elaps_t'], hg['f'] - fg, color=color[0], label='GD')
plt.semilogy(hn['elaps_t'], hn['f'] - fn, color=color[1], label='Newton')
plt.semilogy(hnc['elaps_t'], hnc['f'] - fnc,  color=color[2], label='NCG')
plt.xlabel("time")
plt.ylabel("function values")
plt.title("Function value (time)")
plt.grid()
plt.legend(loc='best')
plt.show()

