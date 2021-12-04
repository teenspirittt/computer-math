import matplotlib.pyplot as plt
import numpy as np

BEGIN, END, INC = -10, 10, 0.001
x = np.arange(BEGIN, END, INC)

NEWTON_POINTS = [-8, -5, 0, 1]
NEWTON_COEFFICIENTS = [2.9, -1.483, 0.237, -4.24]


def f(x):
    return np.tan(0.3 * x + 0.5)


def d1f(x):
    return 0.3 / (np.cos(0.3 * x + 0.5) ** 2)


def d2f(x):
    return 0.18 * np.sin(0.3 * x + 0.5) / np.cos(0.)


def d3f(x):
    return ((0.162 * np.sin(0.3 * x + 0.5) ** 2) / np.cos(0.3 * x + 0.5) ** 4) + (0.054 / np.cos(0.3 * x + 0.5) ** 2)


def t1(x, a):
    return f(a) + d1f(a) * (x - a)


def t2(x, a):
    return t1(x, a) + (d2f(a) * (x - a) ** 2) / 2


def t3(x, a):
    return t2(x, a) + (d3f(a) * (x - a) ** 3) / 6


def m1(x, a):
    return f(x) - t1(x, a)


def m2(x, a):
    return f(x) - t2(x, a)


def m3(x, a):
    return f(x) - t3(x, a)


def get_mistake1(x, a):
    return f(x) - t1(x, a)


def get_mistake2(x, a):
    return f(x) - t2(x, a)


def get_mistake3(x, a):
    return f(x) - t3(x, a)


def draw_mistakes(p):
    mis1, mis2, mis3 = [], [], []
    x = np.arange(BEGIN, END, INC)
    for i in x:
        mis1.append(get_mistake1(i, p))
        mis2.append(get_mistake2(i, p))
        mis3.append(get_mistake3(i, p))
    plt.title("Графики ошибок для точки разложения x = " + str(p))
    plt.plot(x, mis1, label='Ошибка для полинома 1 степени')
    plt.plot(x, mis2, label='Ошибка для полинома 2 степени')
    plt.plot(x, mis3, label='Ошибка для полинома 3 степени')
    plt.grid()
    plt.xlim(-10, 10)
    plt.ylim(-50, 50)
    plt.legend()
    plt.show()


def newton_poly3(x, newP, newC):
    b0, b1, b2, b3 = newC[0], newC[1], newC[2], newC[3]
    x0, x1, x2 = newP[0], newP[1], newP[2]
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1) + b3 * (x - x0) * (x - x1) * (x - x2)


def draw_newton_poly(newP, newC):
    plt.xlim(-10, 10)
    plt.ylim(-40, 40)
    plt.grid()
   # plt.title("")
    plt.plot(x, f(x), color='red')
    plt.plot(x, newton_poly3(x, newP, newC))
    plt.show()


def sliding_filling(length):
    points = []
    b0, b1, b2 = [], [], []
    for i in range(length + 1):
        points.append(BEGIN + 20 / length * i)

    for i in range(len(points) - 2):
        b0.append(f(points[i]))

        b1.append((f(points[i + 1]) - f(points[i])) / (points[i + 1] - points[i]))

        tc = (f(points[i + 1]) - f(points[i])) / (points[i + 1] - points[i])
        tmp = f(points[i + 2]) - f(points[i]) - tc * (points[i + 2] - points[i])
        b2.append(tmp / ((points[i + 2] - points[i]) * (points[i + 2] - points[i + 1])))

    return points, b0, b1, b2


def sliding_poly(b0, b1, b2, x0, x1, x):
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1)


def draw_sliding_poly(points, b0, b1, b2):
    plt.grid()
    plt.title("Скользящие полиномы (n = " + str(len(points) - 2) + ")")
    for i in range(len(points) - 2):
        xs = np.arange(points[i], points[i + 1], 0.001)

        plt.plot(xs, sliding_poly(b0[i], b1[i], b2[i], points[i], points[i + 1], xs))
    plt.show()


draw_mistakes(-5)
draw_mistakes(-1)
draw_mistakes(1)
draw_mistakes(2)
draw_mistakes(5)
draw_newton_poly(NEWTON_POINTS, NEWTON_COEFFICIENTS)

for i in range(2, 24, 1):
    slidingPoints, b0, b1, b2 = sliding_filling(i)
    draw_sliding_poly(slidingPoints, b0, b1, b2)