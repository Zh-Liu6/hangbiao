# 基于CS的轨迹预测，使用三次样条插值对现有点进行插值，得到拟合曲线，使用最后一段拟合曲线来预测后续点
# 效果不如直接使用库中的cubicspline插值函数
import numpy as np
from matplotlib import pyplot as plt
from math import *
from sympy import symbols, diff, simplify

''' 
定义插值函数为: S_i(x_i) = y_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
'''
def cubic_spline(ls):
    """
    三次样条插值
    :param ls: #  输要插值的点，ls为列表，ls中的元素为元组（tuple）
    :return:
    """
    x_1 = [tuple_[0] for tuple_ in ls]  # 插值点的横坐标
    y_1 = [tuple_[1] for tuple_ in ls]  # 插值点的纵坐标
    delta = np.zeros(len(ls) - 1)
    Delta = np.zeros(len(ls) - 1)


    for i in range(len(ls) - 1):
        delta[i] = ls[i + 1][0] - ls[i][0]
        Delta[i] = ls[i + 1][1] - ls[i][1]

    A = np.zeros([len(ls), len(ls)])
    B = np.zeros(len(ls))

    # n个方程求解n个未知变量c_i

    # 列出系数矩阵A
    for i in range(len(ls)):
        if i == 0:
            A[i][0] = 1
            continue
        if i == len(ls) - 1:
            A[i][i] = 1
            continue
        A[i][i - 1] = delta[i - 1]
        A[i][i] = 2 * (delta[i - 1] + delta[i])
        A[i][i + 1] = delta[i]

    # 常数矩阵B
    for i in range(len(ls)):
        if i == 0:
            B[i] = 0
            continue
        if i == len(ls) - 1:
            B[i] = 0
            continue
        B[i] = 3 * (Delta[i] / delta[i] - Delta[i - 1] / delta[i - 1])

    # 求解线性方程组，得到系数c_i
    c = np.linalg.solve(A, B)   # A*x=b,np.linalg.solve求解x

    b = np.zeros(len(ls))
    d = np.zeros(len(ls))

    # 求解b_i,d_i
    for i in range(len(ls) - 1):
        b[i] = Delta[i] / delta[i] - delta[i] / 3 * (2 * c[i] + c[i + 1])
        d[i] = (c[i + 1] - c[i]) / (3 * delta[i])

    # 绘制散点图
    plt.scatter(x_1, y_1, color='blue')
    for i in range(len(ls) - 1):
        x = np.linspace(x_1[i], x_1[i + 1], 100)
        y = y_1[i] + b[i] * (x - x_1[i]) + c[i] * (x - x_1[i]) ** 2 + d[i] * (x - x_1[i]) ** 3
        plt.plot(x, y, color='green')
    plt.show()

    # 求各段的函数
    for i in range(len(ls) - 1):
        x = symbols('x')
        y = y_1[i] + b[i] * (x - x_1[i]) + c[i] * (x - x_1[i]) ** 2 + d[i] * (x - x_1[i]) ** 3
        simplified_y = simplify(y)
        print(simplified_y)

    # 使用最后一段函数来预测
    y = y_1[-2] + b[-2] * (x - x_1[-2]) + c[-2] * (x - x_1[-2]) ** 2 + d[-2] * (x - x_1[-2]) ** 3
    y = simplify(y)
    y_pre = y.subs(x, 5)
    print(y_pre)



if __name__ == '__main__':
    #  输入需要插值的点
    # ls = [(-1.00, -14.58), (0.00, 0.00), (1.27, 0.00), (2.55, 0.00), (3.82, 0.00), (4.92, 0.88), (5.02, 11.17)]
    # 正弦函数三次样条插值
    # ls = [(0, 0), (radians(30), 0.5), (radians(90), 1), (radians(135), 0.5), (radians(180), 0), (radians(225), -0.5), (radians(270), -1), (radians(315), -0.5), (radians(360), 0)]
    ls = [(1, 1), (2, 8), (2.5, 15.625), (3, 27), (4, 64)]
    cubic_spline(ls)



