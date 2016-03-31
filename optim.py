__author__ = 'mikhail'

import scipy.linalg as sp
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import scipy.sparse as sps
from scipy.optimize.linesearch import line_search_wolfe2,line_search_armijo

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
        hist = {'it': np.array([]), 'norm_r': np.array([])}
        hist['it'] = np.append(hist['it'], k)
        hist['norm_r'] = np.append(hist['norm_r'], nev)

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
            hist['norm_r'] = np.append(hist['norm_r'], nev)

        if nev <= tol:
            status = 0
            break

    if trace:
        return xk, status, hist
    else:
        return xk, status



def gd(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.1, disp=False, trace=False):
    start_time = time.time()
    status, x = 1, np.copy(x0)
    f_0 = (lambda x: func(x)[0])
    f_1 = (lambda x: func(x)[1])
    iter = 0

    f, gradient = func(x)
    n_act_evals = 1
    gradient_norm = norm(gradient, np.inf)

    if disp:
        print("%10s %15s %15s %15s" % ('iter', 'oracul_coal', 'func', 'gradient_norm'))
        print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))

    if trace:
        hist = {'f': np.array([]), 'norm_g': np.array([]), 'n_evals': np.array([]), 'elaps_t': np.array([])}
        hist['f'] = np.append(hist['f'], f)
        hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
        hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
        hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)

    for k in range(max_iter):
        if n_act_evals >= max_n_evals:
            break
        if gradient_norm < tol:
            status = 0
            break
        direction = -gradient
        alpha = line_search_wolfe2(f=f_0, myfprime=f_1, xk=x, pk=direction, c1=c1, c2=c2)
        if alpha[0] is None:
            alpha = line_search_armijo(f=f_0, xk=x, pk=direction, gfk=gradient, old_fval=f, c1=c1)

        x = x + alpha[0] * direction

        f, gradient = func(x)
        n_act_evals = n_act_evals + 1
        iter = iter + 1

        gradient_norm = norm(gradient, np.inf)
        if disp:
            print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))
        if trace:
            hist['f'] = np.append(hist['f'], f)
            hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
            hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
            hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)

    if trace:
        return x, f, status, hist
    else:
        return x, f, status


def newton(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.1, disp=False, trace=False):
    start_time = time.time()
    status, x = 1, np.copy(x0)
    f_0 = (lambda x: func(x)[0])
    f_1 = (lambda x: func(x)[1])

    iter = 0

    f, gradient, hessian = func(x)
    n_act_evals = 1
    gradient_norm = norm(gradient, np.inf)

    if disp:
        print("%10s %15s %15s %15s" % ('iter', 'oracul_coal', 'func', 'gradient_norm'))
        print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))

    if trace:
        hist = {'f': np.array([]), 'norm_g': np.array([]), 'n_evals': np.array([]), 'elaps_t': np.array([])}
        hist['f'] = np.append(hist['f'], f)
        hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
        hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
        hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)


    for k in range(max_iter):
        if n_act_evals >= max_n_evals:
            break
        if gradient_norm < tol:
            status = 0
            break


        B = sp.cho_factor(hessian, lower=True, overwrite_a=True)
        direction = sp.cho_solve(B, -gradient, overwrite_b=True)

        alpha = line_search_wolfe2(f=f_0, myfprime=f_1, xk=x, pk=direction, c1=c1, c2=c2)
        if alpha[0] is None:
            alpha = line_search_armijo(f=f_0, xk=x, pk=direction, gfk=gradient, old_fval=f, c1=c1)


        x = x + alpha[0] * direction

        f, gradient, hessian = func(x)
        gradient_norm = norm(gradient, np.inf)
        n_act_evals = n_act_evals + 1
        iter = iter + 1
        if disp:
            print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))

        if trace:
            hist['f'] = np.append(hist['f'], f)
            hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
            hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
            hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)


    if trace:
        return x, f, status, hist
    else:
        return x, f, status


def ncg(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.1, disp=False, trace=False):
    status, x, start_time = 1, np.copy(x0), time.time()
    f_0 = (lambda x: func(x)[0])
    f_1 = (lambda x: func(x)[1])

    f, gradient = func(x)
    direction = -gradient
    n_act_evals = 1
    iter = 0
    gradient_norm = norm(gradient, np.inf)

    if disp:
        print("%10s %15s %15s %15s" % ('iter', 'oracul_coal', 'func', 'gradient_norm'))
        print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))
    if trace:
        hist = {'f': np.array([]), 'norm_g': np.array([]), 'n_evals': np.array([]), 'elaps_t': np.array([])}
        hist['f'] = np.append(hist['f'], f)
        hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
        hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
        hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)


    for k in range(max_iter):
        if n_act_evals >= max_n_evals:
            break
        if norm(gradient, np.inf) < tol:
            status = 0
            break
        alpha = line_search_wolfe2(f=f_0, myfprime=f_1, xk=x, pk=direction, c1=c1, c2=c2)
        if alpha[0] is None:
            alpha = line_search_armijo(f=f_0, xk=x, pk=direction, gfk=gradient, old_fval=f, c1=c1)

        x = x + alpha[0] * direction

        f, gradient_k = func(x)
        n_act_evals =n_act_evals + 1
        betta_k = (gradient_k.dot(gradient_k - gradient))/gradient.dot(gradient)
        direction = -gradient_k + betta_k * direction
        gradient = gradient_k
        gradient_norm = norm(gradient, np.inf)
        iter = iter + 1
        if disp:
            print("%10d %15d %15e %15e" % (iter, n_act_evals, f, gradient_norm))
        if trace:
            hist['f'] = np.append(hist['f'], f)
            hist['norm_g'] = np.append(hist['norm_g'], gradient_norm)
            hist['n_evals'] = np.append(hist['n_evals'], n_act_evals)
            hist['elaps_t'] = np.append(hist['elaps_t'], time.time() - start_time)




    if trace:
        return x, f, status, hist
    else:
        return x, f, status











#def newton(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.9,
#disp=False, trace=False):



#A = np.array([[1, 0], [0, 2]])
#b = np.array([1, 6])
#c = 9.5
#x0 = np.array([0, 0])

#func = (lambda x: ((1/2)*x.dot(A.dot(x)) - b.dot(x) + c, A.dot(x) - b))
#func_hess = (lambda x: ((1/2)*x.dot(A.dot(x)) - b.dot(x) + c, A.dot(x) - b, A))


#set_x, func, status, hist = gd(func, x0, disp=True, trace=True)




#print(set_x, func, status)

def draw_plot(k, n, s):
    color = ['b', 'r', 'g']
    matvec = (lambda x: A.dot(x))

    c = 0
    for i in k:
        for sch in range(0, s):
            B = np.random.randn(n)
            X0 = np.zeros(n)
            V = sp.orth(np.random.randn(n, n))
            AL = sps.diags(np.random.uniform(1, i, n), 0)
            A = np.dot(V, AL.dot(V.T))
            x, r, l = cg(matvec, B, X0, disp=True, trace=True)
            nev = norm(matvec(x) - B)
            plt.semilogy(l['it'], l['norm_r'] - nev, color[c], label='CG '+str(i), alpha=0.7)
            plt.xlabel('it')
            plt.ylabel('norm_r')
            plt.legend(loc='best')
        c = c + 1
    plt.show()


#draw_plot([10, 1000, 100000], 1000, 4)



