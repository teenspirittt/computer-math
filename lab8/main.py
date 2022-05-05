from re import X
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import math

a, b = 1.3, 3.1
e = 0.001
a2, b2 = 0.001, 90
s = 1.763
piece = 10
linear_constrain = LinearConstraint([2, 3], -6, -6)


def f(x):
    return x - np.log(np.log(x))


def df(x):
    return 1 - 1 / (x * np.log(x))


def f2(p):
    x1, x2 = p
    return -1 * (-8 * x1 ** 2 + 4 * x1 - x2 ** 2 + 12 * x2 - 7)


def ff(p):
    x1, x2 = p
    return (2 * x1 - 3 * x2 + 6) ** 2


def chord(start, end, eps):
    x = start
    a = start
    b = end
    n = 0
    x_arr = []
    if df(a) * df(b) > 0:
        return None, None, None
    while abs(df(x)) > eps:
        x = b - df(b) * (b - a) / (df(b) - df(a))
        x_arr.append(abs(x - s))
        if df(a) * df(x) < 0:
            b = x
        else:
            if df(x) * df(b) < 0:
                a = x
        n += 1
    return x, n, x_arr


def barier(params, m):
    return f2(params) + m * ff(params)


def twoD_sol(x1, m, b, eps):
    while m * f2(x1) > eps:
        m = b * m
        result = optimize.minimize(barier, x1, m)
        x1 = result.x
        # print(m * fine_func(x1))
    print(x1)
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = f2([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(X, Y, f([X, Y]), c=f([X, Y ]))
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    plt.title('3D func graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def two_sol():
    initial_guess = [0, 0]
    result = optimize.minimize(f2, initial_guess)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)


def simple_chord():
    X = np.arange(a, b, e)
    x1, n, n_arr = chord(a, b, e)
    plt.plot(X, f(X), label='function graph')
    print(x1, n)
    plt.scatter(x1, f(x1), label='minimize by chord method')
    plt.scatter(s, f(s), label='true minimum')
    plt.grid()
    plt.legend()
    plt.show()
    N = np.arange(0, n, 1)
    plt.plot(N, n_arr, label='error on ones iteration')
    plt.xlabel("iteration")
    plt.ylabel("abs error")
    plt.grid()
    plt.legend()
    plt.show()


def global_chord():
    sol = []
    sol_y = []
    y = []
    h = (b2 - a2) / piece
    for i in range(piece):
        x1 = fibonacci(a2 + i * h, a2 + (i + 1) * h)
        if x1 > 0:
            print(x1)
            sol.append(x1)
            sol_y.append(f(x1))
            print(x1)
    X = np.arange(a2, b2, e)
    for i in range(len(X)):
        y.append(f(X[i]))
    plt.plot(X, y, label='function graph')
    plt.scatter(sol, sol_y, label='minimize by chord method', color='red')
    plt.scatter(s, f(s), label='true minimum')
    plt.grid()
    plt.legend()
    plt.show()


def fibo(n):
    s5 = np.sqrt(5)
    phi = (s5 + 1) / 2
    return int(phi ** n / s5 + 0.5)


def fibonacci(start, end):
    n = 10
    for i in range(n):
        lam = start + (fibo(n - i - 1) / fibo(n - i + 1)) * (end - start)
        mu = start + (fibo(n - i) / fibo(n - i + 1)) * (end - start)
        if f(lam) < f(mu):
            end = mu
        else:
            start = lam
    x_min = (start + end) / 2
    return x_min


simple_chord()
global_chord()
two_sol()
twoD_sol([0, 0], 0.001, 6, 1e-3)
twoD_sol([1, 0], 0.001, 6, 1e-5)
twoD_sol([2, 0], 0.001, 6, 1e-7)
