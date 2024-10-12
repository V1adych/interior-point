import os
import sys
from scipy.optimize import linprog
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interior_point import interior_point


def test_simplex_base():
    A = np.array(
        [
            [130, 100, 155, 85, 50],
            [0.004, 0.005, 0.006, 0.003, 0.004],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.array([200, 0.01, 0.6, 0.6, 0.6, 0.2, 0.05], dtype=np.float32)
    c = np.array([200, 160, 260, 150, 400], dtype=np.float32)

    A_slack = np.hstack([A, np.eye(A.shape[0])])
    c_slack = np.concatenate([c, np.zeros(A.shape[0])])

    _, x_result, _ = interior_point(A_slack, b, c_slack)
    x_test = linprog(-c, A, b).x

    np.testing.assert_array_almost_equal(x_result[: c.shape[0]], x_test, decimal=4)


def test_simplex_custom_1():
    A = np.array(
        [
            [100, 200, 200, 60, 50],
            [4, 5, 6, 3, 4],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.array([130, 4, 6, 6, 6, 2, 0.5], dtype=np.float32)
    c = np.array([130, 200, 25, 100, 430], dtype=np.float32)

    A_slack = np.hstack([A, np.eye(A.shape[0])])
    c_slack = np.concatenate([c, np.zeros(A.shape[0])])

    _, x_result, _ = interior_point(A_slack, b, c_slack)
    x_test = linprog(-c, A, b).x

    np.testing.assert_array_almost_equal(x_result[: c.shape[0]], x_test, decimal=4)


def test_simplex_custom_2():
    A = np.array(
        [[1, 1, 1]],
        dtype=np.float32,
    )
    b = np.array([8], dtype=np.float32)
    c = -np.array([1, 2, 0], dtype=np.float32)

    _, x_result, _ = interior_point(A, b, c)
    x_test = linprog(c=-c, A_eq=A, b_eq=b).x

    np.testing.assert_array_almost_equal(x_result, x_test, decimal=4)
