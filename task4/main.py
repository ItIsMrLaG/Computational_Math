from dataclasses import dataclass
from itertools import product
from math import sqrt, sin, cos

import numpy as np
from scipy.integrate import quad

"""
algorithm from http://www.ict.nsc.ru/matmod/files/textbooks/KhakimzyanovCherny-2.pdf

* chapter 8 (page 78)
** algorithm on page 93
"""


@dataclass
class TridiagMatrix:
    a: np.ndarray  # (i-1) diagonal
    b: np.ndarray  # (i) diagonal
    c: np.ndarray  # (i+1) diagonal


@dataclass
class Problem:
    h: float
    N: int
    x_knots: np.ndarray
    y_coef: np.ndarray  # y_j
    lambd: float


def thomas_algorithm(mt: TridiagMatrix, d: np.ndarray) -> np.ndarray:
    # description: https://gist.github.com/vuddameri/75212bfab7d98a9c75861243a9f8f272

    n: int = d.size
    c1: np.ndarray = np.zeros(n, dtype='float64')
    d1: np.ndarray = np.zeros(n, dtype='float64')
    x: np.ndarray = np.zeros(n + 1, dtype='float64')

    helper = lambda i: mt.b[i] - mt.a[i] * c1[i - 1]
    c_new = lambda i: (mt.c[i] / mt.b[i]) if i == 0 else (mt.c[i] / helper(i))
    d_new = lambda i: (d[i] / mt.b[i]) if i == 0 else (d[i] - mt.a[i] * d1[i - 1]) / helper(i)

    for i in range(n):
        c1[i] = c_new(i)
        d1[i] = d_new(i)

    x_new = lambda i: d1[i] if i == (n - 1) else d1[i] - c1[i] * x[i + 1]

    for i in range(n - 1, -1, -1):
        x[i] = x_new(i)

    return x


def phi_j(p: Problem, j: int, x: float) -> float:
    # page 92

    if j == 0:
        if 0 <= x <= p.x_knots[1]:
            return float(p.x_knots[1] - x) / p.h
        else:
            return 0
    elif j == p.N:
        if p.x_knots[p.N - 1] <= x <= p.x_knots[p.N]:
            return float(x - p.x_knots[p.N - 1]) / p.h
        else:
            return 0
    else:
        if p.x_knots[j - 1] <= x <= p.x_knots[j]:
            return float(x - p.x_knots[j - 1]) / p.h
        elif p.x_knots[j] <= x <= p.x_knots[j + 1]:
            return float(p.x_knots[j + 1] - x) / p.h
        else:
            return 0


"""
    b_j * y_{j-1} + (phi_j, phi_j)_A * y_j + b_{j+1} * y_{j+1} = (f, phi_j)
    |^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   |^^^^^^^^^
    | Left_side                                                  | Right side
"""


def get_left_side_coefs(p: Problem, j: int, k: int) -> float:
    xk: np.ndarray = p.x_knots
    lb: float = p.lambd
    if j > k:
        j, k = k, j

    """ Redefine for wolfram:
        i := h
        a := λ
        m := x[i-1]
        b := x[i]
        c := x[i+1]
    """
    if j == k:  # get (phi_j, phi_j)_A
        # https://www.wolframalpha.com/input?i=%281%2F%28i%5E2%29%29+*%28%28%28integrate+%5B1%2Ba*%28x-m%29%5E2%5D+dx+from+m+to+b%29%29%2B+%28integrate+%5B1%2Ba*%28c-x%29%5E2%5D+dx+from+b+to+c%29%29&assumption=%22i%22+-%3E+%22Variable%22
        return (xk[j + 1] + lb * xk[j] ** 2 * xk[j + 1] - lb * xk[j] * xk[j + 1] ** 2 + (lb * xk[j + 1] ** 3) / 3 -
                xk[j - 1] - lb * xk[j] ** 2 * xk[j - 1] + lb * xk[j] * xk[j - 1] ** 2 - (lb * xk[j - 1] ** 3) / 3) / (
                p.h ** 2)
    elif j + 1 == k:  # get b_j or get b_{j+1}
        # https://www.wolframalpha.com/input?i=%28integrate+%5B-1+%2B+a*%28x-b%29*%28c-x%29%5D+dx+from+b+to+c%29
        return (-1 / 6.0) * (-6 + lb * (xk[j] - xk[j - 1]) ** 2) * (xk[j] - xk[j + 1]) / (p.h ** 2)
    else:
        return 0


def get_right_side_coefs(p: Problem, j: int) -> float:
    xk: np.ndarray = p.x_knots
    lb_sqrt = sqrt(p.lambd)

    """ Redefine for wolfram:
        i := h
        a := λ
        m := x[i-1]
        b := x[i]
        c := x[i+1]
    """
    # get (f, phi_j)
    # https://www.wolframalpha.com/input?i=%281%2F%28i%5E1%29%29+*%28%28%28integrate+%5B%28x-m%29*%282*a+*+sin%28sqrt%28a%29*x%29%29%5D+dx+from+m+to+b%29%29%2B+%28integrate+%5B%28c-x%29*%282*a*+sin%28sqrt%28a%29*x%29%29%5D+dx+from+b+to+c%29%29&assumption=%22i%22+-%3E+%22Variable%22
    first_integral = 2 * (- lb_sqrt * (xk[j] - xk[j + 1]) * cos(lb_sqrt * xk[j]) + sin(lb_sqrt * xk[j]) - sin(
        lb_sqrt * xk[j + 1]))
    second_integral = 2 * (- lb_sqrt * (xk[j] - xk[j - 1]) * cos(lb_sqrt * xk[j]) + sin(lb_sqrt * xk[j]) - sin(
        lb_sqrt * xk[j - 1]))

    return float(first_integral + second_integral) / p.h


def function(p: Problem, x: float) -> float:
    # l_idx, r_idx = 0, p.N - 1
    l_idx, r_idx = 0, p.N

    # get l_idx := i and r_idx := i+1 (when x in [x_i ; x_{i+1}])
    while r_idx - l_idx > 1:
        idx = (r_idx + l_idx) // 2
        if x > p.x_knots[idx]:
            l_idx = idx
        else:
            r_idx = idx

    """
        if x in [x_i ; x_{i+1}] then 
        y_k := sum_{j=1}^{N-1} (y_j * phi_j(x)) = 0 + ... + 0 + y_j * phi_j(x) + y_{j+1} * phi_{j+1}(x) + 0 + ... + 0
    """
    return float(p.y_coef[l_idx]) * phi_j(p, l_idx, x) + float(p.y_coef[r_idx]) * phi_j(p, r_idx, x)


def solve_y(p: Problem):
    A: TridiagMatrix = TridiagMatrix(np.zeros(p.N), np.zeros(p.N), np.zeros(p.N))
    d: np.ndarray = np.zeros(p.N)

    for i in range(1, p.N + 1):
        j = i - 1

        if i >= 1:
            A.a[j] = get_left_side_coefs(p, i - 1, i)

        A.b[j] = get_left_side_coefs(p, j, j)

        if i < p.N:
            A.c[j] = get_left_side_coefs(p, i, i + 1)

        d[j] = get_right_side_coefs(p, j)

    y = thomas_algorithm(A, d)
    y[0] = 0
    y[p.N] = 0
    p.y_coef = y


"""
    h: float
    N: int
    x_knots: np.ndarray
    y_coef: np.ndarray  # y_j
    lambd: float
"""


def solve_problem(l: float, lambd: float, n: int) -> Problem:
    x: np.ndarray = np.linspace(0.0, l, n)
    p = Problem(
        h=float(x[1]) - float(x[0]),
        x_knots=x,
        N=n - 1,
        y_coef=np.zeros(1),
        lambd=lambd
    )
    solve_y(p)
    return p


def eval_error(p: Problem, l: float) -> tuple[float, float]:
    f = lambda x: 2 * p.lambd * np.sin(np.sqrt(p.lambd) * x)

    test_n: int = p.N * 10
    test_x: np.ndarray = np.linspace(0.0, l, test_n)

    test_h2: float = (float(test_x[1]) - float(test_x[0])) ** 2

    # ||f||_{L_2(0;l)}
    norm_f = np.sqrt(quad(lambda x: f(x) ** 2, float(test_x[0]), float(test_x[-1]))[0])

    approx_val: np.ndarray = np.array([function(p, xk) for xk in test_x])
    real_val: np.ndarray = np.array([sin(sqrt(p.lambd) * xk) for xk in test_x])

    # || y - y_k ||_{L_2(0;l)}
    err = np.sqrt(np.sum((real_val - approx_val) ** 2))

    """
        c := 1/c_1 * ((Q*l/2 + P_1) * l/(2*c_1) + 1)
        
        P_1 := max|p'(x)| ; x in [0, l]
        Q := max(q(x)) ; x in [0, l]
        c_1 := const > 0 
    """
    c = (p.lambd * (l ** 2) / 4) + 1

    """
        c' := J_M * np.sqrt(P + Q * l ** 2 / 4)
        
        P := max(p(x)) ; x in [0, l]
        Q := max(q(x)) ; x in [0, l]
        J_M := const > 0 
    """
    J_M = 0.1
    c_ = J_M * np.sqrt(1 + p.lambd * l ** 2 / 4)

    return err, (c * c_) ** 2 * p.h ** 2 * norm_f


@dataclass
class TestCase:
    h_2: float
    err: float
    N: int
    lambd: float


if __name__ == '__main__':
    test_set = list(product([1, 10, 100, 1000], [10, 20, 1000, 10000]))
    results: list[TestCase] = []

    for lambd, N in test_set:
        l: float = 16 * np.pi / sqrt(lambd)
        p: Problem = solve_problem(l, lambd, N)

        err, h = eval_error(p, l)

        results.append(TestCase(
            h_2=h,
            err=err,
            N=N,
            lambd=lambd
        ))
    t = 0
    f = 0
    for el in results:
        # print(f"{el.err}, {el.h_2}, {el.N}, {el.lambd}")
        if el.err < el.h_2:
            t += 1
        else:
            f += 1
            print(f"{el.N}, {el.lambd}")
    print(f"true = {t}, false = {f}")
