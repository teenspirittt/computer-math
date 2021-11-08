import math
import numpy as np
import sympy as sp


def iter(f, f0, x, eps):
    it = 0
    while np.linalg.norm(x - f(x)) >= eps:
        it += 1
        x = f(x)
    return x, f0(x), it


def seidel(f, f0, x, eps):
    it = 0
    n = len(x)
    while np.linalg.norm(x - f(x)) >= eps:
        it += 1
        for i in range(n):
            x[i] = f(x)[i]
    return x, f0(x), it


def newton(f, j, x, eps):
    it = 0
    x_next = f(x)
    while np.linalg.norm(x - x_next) >= eps:
        it += 1
        x = x_next
        w_inv = np.linalg.inv(j(x))
        x_next = x - w_inv.dot(f(x))
    return x_next, f(x), it


x_sym = sp.symbols("x")
y_sym = sp.symbols("y")
f1_sym = sp.sin(x_sym + y_sym) - 1.4 * x_sym
f2_sym = x_sym ** 2 + y_sym ** 2 - 1
df1dx = sp.diff(f1_sym, x_sym)
df1dy = sp.diff(f1_sym, y_sym)
df2dx = sp.diff(f2_sym, x_sym)
df2dy = sp.diff(f2_sym, y_sym)


def f_1x(x):
    return (math.sin(x[0] + x[1])) / 1.4


def f_2y(x):
    return abs(1.0 - x[0] ** 2) ** 0.5


def f1(x):
    return math.sin(x[0] + x[1]) - 1.4 * x[0]


def f2(x):
    return x[0] ** 2 + x[1] ** 2 - 1


def fn(x):
    return np.array([f1(x), f2(x)])


def fx_fy(x):
    return np.array([f_1x(x), f_2y(x)])


def f1y(x):
    return -x - np.sin(1.4 * x) ** (-1)


def f2y(x):
    return abs(1.0 - x ** 2) ** 0.5


def jacob(x):
    return np.array([
        [float(df1dx.subs(x_sym, x[0]).subs(y_sym, x[1]).evalf()),
         float(df1dy.subs(x_sym, x[0]).subs(y_sym, x[1]).evalf())],
        [float(df2dx.subs(x_sym, x[0]).subs(y_sym, x[1]).evalf()),
         float(df2dy.subs(x_sym, x[0]).subs(y_sym, x[1]).evalf())]])


def res_print(in_tup):
    (x, err, it) = in_tup
    print("x,y = %s \nКоличество итераций = %d " % (x, it))


def calc(eps, x0):
    print("========================================")
    print("e = ", "{:.0e}".format(eps), "\nx0 = %s\n" % x0)
    print("Метод итераций: ")
    res_print(iter(fx_fy, fn, x0.copy(), eps))
    print("\nМетод Зейделя: ")
    res_print(seidel(fx_fy, fn, x0.copy(), eps))
    print("\nМетод Ньютона: ")
    res_print(newton(fn, jacob, x0.copy(), eps))
    print("========================================\n")


calc(x0=np.array([0.8, 0.6]), eps=0.0001)
calc(x0=np.array([-1., -0.3]), eps=0.0001)
calc(x0=np.array([-0.1, -0.1]), eps=0.0001)
#calc(x0=np.array([10., 1.]), eps=0.0001)
