import matplotlib.pyplot as plt
import numpy as np


# Первоначальная функция
def f(x):
    return np.tan(0.3 * x + 0.5)


# Первая производная
def d1f(x):
    return 0.3 / (np.cos(0.3 * x + 0.5) ** 2)


# Вторая производная
def d2f(x):
    return 0.18 * np.sin(0.3 * x + 0.5) / np.cos(0.3 * x + 0.5) ** 3


# Третья производная
def d3f(x):
    return ((0.162 * np.sin(0.3 * x + 0.5) ** 2) / np.cos(0.3 * x + 0.5) ** 4) + (0.054 / np.cos(0.3 * x + 0.5) ** 2)


# Разложение по Тейлору 1
def t1f(x, a):
    return f(x) + d1f(x) * (x - a)


# Разложение по Тейлору 2
def t2f(x, a):
    return t1f(x, a) + d2f(x) * (x - a) ** 2 / 2


# Разложение по Тейлору 3
def t3f(x, a):
    return t2f(x, a) + d3f(x) * (x - a) ** 3 / 6


def getMistake1(x, a):
    return f(x) - t1f(x, a)


def getMistake2(x, a):
    return f(x) - t2f(x, a)


def getMistake3(x, a):
    return f(x) - t3f(x, a)


def drawMistakes(p):
    m1, m2, m3 = [], [], []
    x = np.arange(-10, 10, 0.001)
    for i in x:
        m1.append(getMistake1(i, p))
        m2.append(getMistake2(i, p))
        m3.append(getMistake3(i, p))
    plt.plot(x, m1, label='m1')
    plt.plot(x, m2, label='m2')
    plt.plot(x, m3, label='m3')
    plt.grid()
    plt.xlim(-10, 10)
    plt.ylim(-50, 50)
    plt.legend()
    plt.show()


def getNewton3(x, xp, a):
    return a[0] + a[1] * (x - [xp[0]]) + a[2] * (x - xp[0]) * (x - xp[1]) + a[3] * (x - xp[0]) * (x - xp[1]) * (
        x - xp[2])


def drawNewton():
    xp3 = [-8, -5, 0, 1]
    a3 = [2.9, -1.483, 0.237, -4.24]

    x = np.arange(-10, 10, 0.1)
    m3, p = [], []
    for i in x:
        m3.append(getNewton3(i, xp3, a3))
    plt.plot(x, m3, label='Newton (3) mistake ')
    plt.grid()
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()


drawMistakes(-5)
drawMistakes(-1)
drawMistakes(1)
drawMistakes(2)
drawMistakes(5)
drawMistakes(7)
drawMistakes(8)
drawMistakes(9)
drawNewton()
