import numpy as np
import random as rn
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

A = 6
B = 10
N = 40
STEP = (B - A) / N


def f(x):
    return np.log(x) ** np.sin(x)


def noise(x):
    return f(x) + rn.uniform(-f(x) / 20, f(x) / 20)


def smooth(X, Y):
    n = len(Y) - 1
    y = np.full(len(Y), 0.)
    y[0] = (5 * Y[0] + 3 * Y[1] + Y[2] + Y[3] + Y[4]) / 11
    y[1] = (3 * Y[0] + 5 * Y[1] + 3 * Y[2] + 2 * Y[3] + Y[4]) / 14
    y[n - 1] = (3 * Y[n] + 5 * Y[n - 1] + 3 *
                Y[n - 2] + 2 * Y[n - 3] + Y[n - 2]) / 14
    y[n] = (5 * Y[n] + 3 * Y[n - 1] + Y[n - 2] + Y[n - 3] + Y[n - 4]) / 11
    for i in range(2, n - 1):
        y[i] = (Y[i - 2] + Y[i - 1] + Y[i] + Y[i + 1] + Y[i + 2]) / 5
    fx = np.arange(A, B, 0.01)
    fy = f(fx)
    f5 = plt.figure()
    plt.plot(fx, fy, label="Original graphic", linestyle="--", color="black")
    plt.plot(X, y, label="Smoothed line", color="red")
    plt.scatter(X, Y, label="Noised points", color="green")
    plt.grid()
    plt.legend()
    plt.show()
    f5.savefig("images/lin_smooth.png")
    return y


def spline(X, Y):
    fx = np.arange(A, B, 0.1)
    sy = CubicSpline(X, Y)
    f3 = plt.figure()
    plt.plot(fx, f(fx), label="Original line", color="black")
    plt.plot(fx, sy(fx), label="Spline", color="red")
    plt.grid()
    plt.legend()
    plt.show()
    mistake(fx, sy(fx), "Spline mistake")
    f3.savefig("images/spline.png")


def ls_poly(cs, x):
    sum = 0
    for i in range(len(cs)):
        sum += cs[i] * x ** i
    return sum


def least_squaresS(X, Y, m):
    xy = []
    for k in (range(len(X) - 1)):
        xy.append([X[k], Y[k]])
    matrix = [[0] * m for i in range(m)]
    right = [0] * m
    for i in range(m):
        for j in range(m):
            s1, s2 = 0, 0
            for k in range(len(xy)):
                s1 += xy[k][0] ** (i + j)
                s2 += xy[k][1] * xy[k][0] ** i
            matrix[i][j] = s1
            right[i] = s2
    cs = np.linalg.solve(matrix, right)
    fx = np.arange(A, B, STEP)
    mistake(fx, ls_poly(cs, fx), "Least squares(" + str(m) + ") mistake")
    f4 = plt.figure()
    plt.plot(fx, f(fx), label="Original line", color="black")
    plt.plot(fx, ls_poly(cs, fx),
             label="Least squares line(" + str(m) + ")", color="red")
    plt.grid()
    plt.legend()
    plt.show()
    f4.savefig("images/least_sq" + str(m) + ".png")


def mistake(X, Y, msg):
    f6 = plt.figure()
    plt.plot(X, f(X) - Y, label=msg, color="black")
    s = 0
    for i in range(len(Y)):
        s += (f(X[i]) - Y[i]) ** 2
    s = np.sqrt(s) / len(Y)
    print(msg + "= " + str(s))
    plt.grid()
    plt.legend()
    plt.show()
    f6.savefig("images/" + msg + ".png")


x = np.arange(A, B, 0.01)
f1 = plt.figure()
plt.plot(x, f(x), label="Original line", color="black")
plt.grid()
plt.legend()
plt.show()
f1.savefig("images/fx.png")

X = np.arange(A, B, 0.01)
Xn = np.arange(A, B + STEP, STEP)
Yn = []
for i in Xn:
    Yn.append(noise(i))
f2 = plt.figure()
plt.scatter(Xn, Yn, color="black")
plt.show()
f2.savefig("images/noised.png")

Yns = smooth(Xn, Yn)
spline(Xn, Yns)

for i in [2, 3, 4, 5, 7, 10, 15, 20]:
    least_squaresS(Xn, Yn, i)
