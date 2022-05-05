import matplotlib.pyplot as plt
from mpmath.functions.functions import re
from numpy.lib.function_base import gradient
import sympy
from sympy.abc import D, r
import numpy as np


def f(x):
    return np.sqrt(x) * np.sin(x)


# Левая разностная производная
def left(m, x, y):
    i = 1
    for j in range(1, len(x)):
        if x[j - 1] <= m < x[j]:
            i = j
            break
    return (y[i] - y[i - 1]) / (x[i] - x[i-1])


# Правая разностная производная
def right(m, x, y):
    i = 0
    for j in range(len(x) - 1):
        if x[j] <= m < x[j + 1]:
            i = j
            break
    return (y[i+1] - y[i] / (x[i + 1]) - x[i])


# Центральная разностная производная
def center(m, x, y):
    i = 0
    for j in range(1, len(x) - 1):
        if x[j - 1] <= m < x[j + 1]:
            i = j
            break
    return 0.5 * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i])


# Приближенное соотношение для второй производной
def sec_dif(m, x, y):
    i = 0
    for j in range(1, len(x) - 1):
        if x[j - 1] <= m < x[j + 1]:
            i = j
            break
    return (y[i - 1] - 2 * y[i] + y[i + 1]) / (x[i + 1] - x[i]) ** 2


x = [0.5, 1.0, 1.5, 2.0]
y = [3.2, 4.5, 2.6, 1.4]
a, b, c = 0.76, 1.2, 1.6

print('Left:')
print('a: ' + str(left(a, x, y)))
print('b: ' + str(left(b, x, y)))
print('c: ' + str(left(c, x, y)))
print('Right:')
print('a: ' + str(right(a, x, y)))
print('b: ' + str(right(b, x, y)))
print('c: ' + str(right(c, x, y)))
print('Center:')
print('a: ' + str(center(a, x, y)))
print('b: ' + str(center(b, x, y)))
print('c: ' + str(center(c, x, y)))
print('Second lvl diff.:')
print('a: ' + str(sec_dif(a, x, y)))
print('b: ' + str(sec_dif(b, x, y)))
print('c: ' + str(sec_dif(c, x, y)))


# --------- 1.1.2 ----------


def der1_f(x):
    return np.sqrt(x) * np.cos(x) + np.sin(x) / 2 * np.sqrt(x)


def der2_f(x):
    return -np.sqrt(x) * np.sin(x) + np.cos(x)/np.sqrt(x) - np.sin(x)/4 * x ** (3/2)


def an_right(x, h):
    return (f(x + h) - f(x)) / h


def an_left(x, h):
    return (f(x) - f(x - h)) / h


def an_center(x, h):
    return 0.5 * (f(x + h) - f(x - h)) / h


def an_sec_dif(x, h):
    return (f(x - h) - 2 * f(x) + f(x + h)) / h ** 2


def draw_dif1(x, h):
    plt.grid()
    plt.title("Graphs for 1'st derivative, h = " + str(h))
    plt.plot(x, an_right(x, h), c='g', label='right defference relation')
    plt.plot(x, an_left(x, h), c='r', label='left defference relation')
    plt.plot(x, an_center(x, h), c='b', label='center defference relation')
    plt.legend()
    plt.show()


def draw_dif2(x, h):
    plt.grid()
    plt.title("Graphs for 2'nd derivative, h = " + str(h))
    plt.plot(x, der2_f(x), c='r', label='2 derivative from table of derivatives')
    plt.plot(x, an_sec_dif(x, h), c='g',
             label='Approximates relation for 2nd derivative ')
    plt.legend()
    plt.show()
    plt.title("Error graph")
    plt.plot(x, abs(der2_f(x) - an_sec_dif(x, h)))
    plt.grid()
    plt.show()


h = [0.5, 0.1, 0.01]
xs = np.arange(0, 2 * np.pi, 0.001)

draw_dif1(xs, h[0])
draw_dif1(xs, h[1])
draw_dif1(xs, h[2])

draw_dif2(xs, h[0])
draw_dif2(xs, h[1])
draw_dif2(xs, h[2])


# --------- 1.2.1 ----------
def f_i(x):
    return x ** 3 * np.cos(x ** 2)


def integral_f(x):
    return -1/2 + np.pi ** 2 * np.sin(np.pi ** 2 / 16) / 32 + np.cos(np.pi ** 2 / 16) / 2


def middle_rectangles(a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(1, n + 1):
        xi = a + h * (i - 0.5)
        integral += f(xi)
    return h * integral


def trapezium(a, b, n):
    integral = (f(a) + f(b)) / 2
    x = a
    h = (b - a) / n
    for i in range(1, n):
        x += h
        integral += f(x)
    return h * integral


def parabola(a, b, n):
    if n % 2:
        n += 1
    integral = f(a) + f(b)
    x = a
    h = (b - a) / n
    for i in range(1, n):
        x += h
        if i % 2:
            integral += 4 * f(x)
        else:
            integral += 2 * f(x)
    return h / 3 * integral


n = [1, 2, 4, 10, 50, 100]
begin, end = np.pi/4, 0


print("Integral value calculated by Newton's method")
print(integral_f(end) - integral_f(begin))

print("Middle rectangles method:")
for i in range(len(n)):
    print("n = " + str(n[i]))
    print(middle_rectangles(begin, end, n[i]))
print("\n")

print("Trapezium rectangles method:")
for i in range(len(n)):
    print("n = " + str(n[i]))
    print(trapezium(begin, end, n[i]))
print("\n")

print("Parabola method:")
for i in range(len(n)):
    print("n = " + str(n[i]))
    print(parabola(begin, end, n[i]))
print("\n")

# --------- 1.2.2 ----------

ans = integral_f(end) - integral_f(begin)
nFinalTrap, nFinalSimp, nFinalRect = [], [], []
ni = np.array([1e-1, 1e-3, 1e-5, 1e-8])
for eps in ni:
    nTrap, nSimp, nRect = 2, 2, 2

    while (abs(trapezium(begin, end, nTrap) - trapezium(begin, end, nTrap - 1))) > eps:
        nTrap += 1
    print("Relation value:" + str(trapezium(begin, end, nTrap)))
    nFinalTrap.append(nTrap)

print("Trapezium method")
print(nFinalTrap)
print("")
