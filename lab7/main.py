import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.integrate import solve_ivp


def func(x, y):
    return x * y ** 2 - y * x - x ** 2


def euler_method(a, b, y0, h):
    x = [a]
    y = [y0]
    for i in range(1, int((b - a) / h) + 1):
        x.append(x[i - 1] + h)
        y.append(y[i - 1] + h * func(x[i - 1], y[i - 1]))
    return y


def implicit_euler_method(a, b, y0, h):
    x = [a]
    y = [y0]
    for i in range(1, int((b - a) / h) + 1):
        x.append(x[i - 1] + h)
        xm = x[i - 1] + h
        ym = y[i - 1] + h * func(x[i - 1], y[i - 1])
        y.append(y[i - 1] + h * (func(x[i - 1], y[i - 1]) + func(xm, ym)) / 2)
    return y


def runge_kutta4_method(a, b, y0, h):
    x = [a]
    y = [y0]
    for i in range(1, int((b - a) / h) + 1):
        x.append(x[i - 1] + h)
        d1 = func(x[i - 1], y[i - 1])
        d2 = func(x[i - 1] + h / 2, y[i - 1] + d1 * h / 2)
        d3 = func(x[i - 1] + h / 2, y[i - 1] + d2 * h / 2)
        d4 = func(x[i - 1] + h, y[i - 1] + d3 * h)
        d_ = (d1 + 2 * d2 + 2 * d3 + d4) / 6
        y.append(y[i - 1] + h * d_)
    return y


def sol_with_scipy(a, b, h):
    x = np.arange(a, b + h, h)
    res = solve_ivp(func, [0, 1], [1], t_eval=x)
    return x, res.y[0]


def draw_graphs(a, b, res, h, lbl):
    xi = np.arange(a, b + h, h)
    sol = sol_with_scipy(a, b, h)
    plt.plot(xi, res, label=lbl, c='r')
    plt.plot(sol[0], sol[1], label="Scipy", c='g')
    plt.plot(xi, res, c='r')
    plt.plot(xi, sol[1], c='g')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title('Error')
    plt.plot(xi, abs(res - sol[1]), label='Error graph ' + lbl)
    plt.legend()
    plt.grid()
    plt.show()
    sum = 0
    for i in range(len(res)):
        sum += np.std([res[i], sol[1][i]])
    print('Error = ', sum, lbl)


x0, xn = 0, 1
y0 = 1

for hi in range(1, 4):
    hi = 10 ** (-hi)
    res1 = euler_method(x0, xn, y0, hi)
    res2 = implicit_euler_method(x0, xn, y0, hi)
    res3 = runge_kutta4_method(x0, xn, y0, hi)

    draw_graphs(x0, xn, res1, hi, "Euler")
    draw_graphs(x0, xn, res2, hi, "Implicit Euler")
    draw_graphs(x0, xn, res3, hi, "Runge-Kutta 4")
