import math


def F(x):
    return math.tan(0.3 * x + 0.5) - x ** 2


def F1(x):
    return -2 * x + (0.3 / math.cos(0.3 * x + 0.5) ** 2)


def G(x):
    return x - F(x) / F1(x)


a1, b1 = -1.0, 0.0
e = 0.0000000001


def half_divide_method(a, b):
    x = (a + b) / 2
    while math.fabs(F(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if F(a) * F(x) < 0 else (x, b)
    return (a + b) / 2


def simple_iteration(a, b):
    if F(a) * F(b) < 0:
        xn = a
        xn1 = b
        while abs(xn1 - xn) > e:
            xn = xn1
            xn1 = G(xn)
        return round(xn1, 4)
    else:
        return "choose another interval"


def chord_tangent_method(a, b):
    x0 = a
    if F(a) * F(b) > 0:
        print('a or b is incorrect')
    else:
        x11 = x0 - F(x0) / F1(x0)
        x12 = a - ((b - a) * F(a) / (F(b) - F(a)))
        e1 = (x11 + x12) / 2
        while abs(e1 - x11) > e:
            a = x11
            b = x12
            x11 = a - F(a) / F1(a)
            x12 = a - ((b - a) * F(a) / (F(b) - F(a)))
            e1 = (x11 + x12) / 2
        return x11


print('root of the equation half divide method \n x = '
      '%.10f' % half_divide_method(a1, b1))

print('root of the equation simple iteration method \n x = '
      '%.10f' % simple_iteration(a1, b1))

print('root of the equation chord tangent method \n x = '
      '%.10f' % chord_tangent_method(a1, b1))
