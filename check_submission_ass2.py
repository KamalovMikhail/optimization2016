import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
import unittest
from ddt import ddt, data, unpack

from io import StringIO
import sys
from special import grad_finite_diff, hess_finite_diff
from lossfuncs import logistic

from optim import cg, gd, newton, ncg

############################################################################################################
# Check if it's Python 3
if not sys.version_info > (3, 0):
    print('You should use only Python 3!')
    sys.exit()

############################################################################################################
######################################### Auxiliary functions ##############################################
############################################################################################################

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

class MyList(list):
    pass

def annotated(min_method, min_func):
    r = MyList([min_method, min_func])
    setattr(r, '__name__', 'test_%s' % min_method.__name__)
    return r

############################################################################################################
############################################# TestLogistic #################################################
############################################################################################################

# Simple data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, -1, 1])
reg_coef = 0.5
w = np.array([0, 0])

class TestLogistic(unittest.TestCase):
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        f, g = logistic(w, X, y, reg_coef)

        self.assertTrue(isinstance(g, np.ndarray))

        assert_almost_equal(f, 0.693, decimal=2)
        assert_array_almost_equal(g, [0, -0.25])

    def test_hess(self):
        """Check that Hessian is returned correctly when `hess=True`."""
        f, g, H = logistic(w, X, y, reg_coef, hess=True)

        self.assertTrue(isinstance(H, np.ndarray))

        assert_array_almost_equal(H, [[0.625, 0.0625], [0.0625, 0.625]])

############################################################################################################
############################################ TestFiniteDiff ################################################
############################################################################################################

# Define a simple quadratic function
A = np.array([[1, 0], [0, 2]])
b = np.array([1, 6])
phi = (lambda x: (1/2)*x.dot(A.dot(x)) + b.dot(x))

class TestFiniteDiff(unittest.TestCase):
    def test_grad_finite_diff(self):
        """Check the function returns a correct gradient."""
        g = grad_finite_diff(phi, np.array([0, 0]))

        self.assertTrue(isinstance(g, np.ndarray))

        assert_array_almost_equal(g, b)

    def test_hess_finite_diff(self):
        """Check the function returns a correct Hessian."""
        H = hess_finite_diff(phi, np.array([0, 0]))

        self.assertTrue(isinstance(H, np.ndarray))

        assert_array_almost_equal(H, A)

############################################################################################################
################################################# TestCG ###################################################
############################################################################################################

# Define a simple linear system with A = A' > 0
A = np.array([[1, 0], [0, 2]])
b = np.array([1, 6])
x0 = np.array([0, 0])
matvec = (lambda x: A.dot(x))

class TestCG(unittest.TestCase):
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_sol, status = cg(matvec, b, x0)

        assert_equal(status, 0)
        self.assertTrue(norm(A.dot(x_sol) - b, np.inf) <= 1e-5)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Try high accuracy."""
        x_sol, status = cg(matvec, b, x0, tol=1e-10)

        assert_equal(status, 0)
        self.assertTrue(norm(A.dot(x_sol) - b, np.inf) <= 1e-10)

    def test_max_iter(self):
        """Check argument `max_iter` is supported and can be set to None."""
        x_sol, status = cg(matvec, b, x0, max_iter=None)

        assert_equal(status, 0)
        self.assertTrue(norm(A.dot(x_sol) - b, np.inf) <= 1e-5)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            cg(matvec, b, x0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_sol, status, hist = cg(matvec, b, x0, trace=True)

        self.assertTrue(isinstance(hist['norm_r'], np.ndarray))

############################################################################################################
############################################### TestOptim ##################################################
############################################################################################################

# Define a simple quadratic function for testing
A = np.array([[1, 0], [0, 2]])
b = np.array([1, 6])
c = 9.5
x0 = np.array([0, 0])
func = (lambda x: ((1/2)*x.dot(A.dot(x)) - b.dot(x) + c, A.dot(x) - b))
func_hess = (lambda x: ((1/2)*x.dot(A.dot(x)) - b.dot(x) + c, A.dot(x) - b, A))
# For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

testing_pairs = (
    annotated(gd, func),
    annotated(ncg, func),
    annotated(newton, func_hess),
)

@ddt
class TestOptim(unittest.TestCase):
    @data(*testing_pairs)
    @unpack
    def test_default(self, min_method, min_func):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, f_min, status = min_method(min_func, x0)

        assert_equal(status, 0)
        self.assertTrue(norm(A.dot(x_min) - b, np.inf) <= 1e-4)
        self.assertTrue(abs(f_min) <= 1e-8)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    @data(*testing_pairs)
    @unpack
    def test_tol(self, min_method, min_func):
        """Try high accuracy."""
        x_min, f_min, status = min_method(min_func, x0, tol=1e-8)

        assert_equal(status, 0)
        self.assertTrue(norm(A.dot(x_min) - b, np.inf) <= 1e-8)
        self.assertTrue(abs(f_min) <= 1e-14)

    @data(*testing_pairs)
    @unpack
    def test_max_iter(self, min_method, min_func):
        """Check if argument `max_iter` is supported."""
        min_method(min_func, x0, max_iter=15)

    @data(*testing_pairs)
    @unpack
    def test_max_n_evals(self, min_method, min_func):
        """Check if the method exceeds the limit on `max_n_evals`."""
        n_act_evals = 0
        def min_func_wrapper(x):
            nonlocal n_act_evals
            n_act_evals += 1
            return min_func(x)

        min_method(min_func_wrapper, x0, max_n_evals=1)

        self.assertTrue(n_act_evals <= 1)

    @data(*testing_pairs)
    @unpack
    def test_c1(self, min_method, min_func):
        """Check if argument `c1` is supported."""
        min_method(min_func, x0, c1=0.2)

    @data(*testing_pairs)
    @unpack
    def test_c2(self, min_method, min_func):
        """Check if argument `c2` is supported."""
        min_method(min_func, x0, c2=0.1)

    @data(*testing_pairs)
    @unpack
    def test_disp(self, min_method, min_func):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            min_method(min_func, x0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    @data(*testing_pairs)
    @unpack
    def test_trace(self, min_method, min_func):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, f_min, status, hist = min_method(min_func, x0, trace=True)

        self.assertTrue(isinstance(hist['f'], np.ndarray))
        self.assertTrue(isinstance(hist['norm_g'], np.ndarray))
        self.assertTrue(isinstance(hist['n_evals'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['norm_g']), len(hist['f']))
        assert_equal(len(hist['n_evals']), len(hist['f']))
        assert_equal(len(hist['elaps_t']), len(hist['f']))

        # make sure hist['n_evals'] is a cumulative sum
        self.assertTrue(np.all(hist['n_evals'] >= 0))
        self.assertTrue(np.all(hist['n_evals'][1:] - hist['n_evals'][:-1] > 0))

############################################################################################################
################################################## Main ####################################################
############################################################################################################

if __name__ == '__main__':
    unittest.main()