__author__ = 'mikhail'

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
import unittest
from ddt import ddt, data, unpack

from io import StringIO
import sys

from optim import cg

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

if __name__ == '__main__':
    unittest.main()