import numpy as np
from pyproj import CRS, Transformer
from math import *
from matplotlib import pyplot as plt
from sympy import symbols, diff
import pandas as pd

def WGS84_to_plane(lon, lat):
    """
    WGS84坐标系转换到高斯坐标系
    :param lon:
    :param lat:
    :return:
    """
    crs_WGS84 = CRS.from_epsg(4326)  # WGS84坐标系
    # 带号为3度带的中央经线经度
    d = 111  # 3度带的中央经线经度
    format = '+proj=tmerc +lat_0=0 +lon_0=' + str(d) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(format)  # 高斯坐标系
    transformer = Transformer.from_crs(crs_WGS84, crs_GK)
    x, y = transformer.transform(lat, lon)  # 将WGS84坐标转换为高斯坐标
    return x, y

def speed_transform(sog, cog):
    """
    速度转换
    :param sog:对地航速，单位：十分之一节
    :param cog:对地航向，单位：十分之一度
    :return: X轴速度分量, Y轴速度分量
    """
    v = 0.5144 * sog / 10
    vx = v * cos(pi * cog / 10 / 180)
    vy = v * sin(pi * cog / 10 / 180)
    return vx, vy

def cubic_spline(ls):
    """
    三次样条插值
    :param ls: 输要插值的点，ls为列表，ls中的元素为元组（tuple）
    :return: 根据三次样条插值得到曲线的表达式，求导得出当前加速度
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

    # # 绘制散点图和函数曲线图
    # plt.scatter(x_1, y_1, color='blue')
    # for i in range(len(ls) - 1):
    #     x = np.linspace(x_1[i], x_1[i + 1], 100)
    #     y = y_1[i] + b[i] * (x - x_1[i]) + c[i] * (x - x_1[i]) ** 2 + d[i] * (x - x_1[i]) ** 3
    #     plt.plot(x, y, color='green')
    # plt.show()

    x = symbols('x')  # 定义符号
    f = y_1[-2] + b[-2] * (x - x_1[-2]) + c[-2] * (x - x_1[-2]) ** 2 + d[-2] * (x - x_1[-2]) ** 3
    # 求一阶导数
    f_prime = diff(f, x)
    # print(f"函数f", f)
    value = f_prime.subs(x, x_1[-1])  # 将x=x_1[-1]带入方程，即求出预测的下一个状态的加速度
    # print(f"在x_point处的导数为{value}")
    return value



class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x0):
        """
        初始化卡尔曼滤波器
        :param F: 状态转移矩阵
        :param B: 控制输入矩阵
        :param H: 观测矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 观测噪声协方差矩阵
        :param P: 初始状态协方差矩阵
        :param x0: 初始状态向量
        """
        self.F = F  # 状态转移矩阵
        self.B = B  # 控制输入矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差矩阵
        self.R = R  # 观测噪声协方差矩阵
        self.P = P  # 初始状态协方差矩阵
        self.x = x0  # 初始状态向量

    def predict(self, u):
        """
        预测步骤
        :param u: 控制输入
        :return: 预测的状态和协方差
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        print()
        return self.x, self.P

    def update(self, z, ax, ay):
        """
        更新步骤
        :param z: 观测值
        :param ax: x方向加速度
        :param ay: y方向加速度
        :return: 更新的状态和协方差
        """
        y = z - np.dot(self.H, self.x)  # 残差
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R  # 残差协方差
        sni = np.linalg.inv(S)  # 求S的逆矩阵
        # K = np.dot(self.P, self.H.T) / S  # 卡尔曼增益
        K = np.dot(self.P, self.H.T).dot(sni)

        self.x = self.x + np.dot(K, y)  # 更新状态
        self.x[4][0] = ax   # 修改加速度
        self.x[5][0] = ay
        self.P = self.P - np.dot(K, self.H).dot(self.P)  # 更新协方差
        return self.x, self.P


if __name__ == '__main__':

    # 系统参数
    T = 60 # 采样时间间隔
    F = np.array([[1, 0, T, 0, T*T/2, 0], [0, 1, 0, T, 0, T*T/2], [0, 0, 1, 0, T, 0], [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])  # 状态转移矩阵，例如 [X,Y位置, X,Y速度, X,Y加速度]
    B = np.array([[0], [0], [0], [0], [0], [0]])  # 控制输入矩阵，例如 加速度影响（此处加速度已经加入状态向量，故B为零）
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])  # 观测矩阵，观测X,Y轴位置及速度
    Q = np.array([[0.1, 0, 0, 0, 0, 0], [0, 0.01, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])  # 过程噪声协方差矩阵
    R = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # 观测噪声协方差矩阵


    P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])  # 初始状态协方差矩阵

    df = pd.read_csv("./data/aistra.csv", header=None, index_col=False)
    lon = df.iloc[:5, 2]
    lat = df.iloc[:5, 3]
    cog = df.iloc[:5, 4]
    sog = df.iloc[:5, 6]
    lon_plane = []
    lat_plane = []
    vx_plane = []
    vy_plane = []


    # 初始参数
    xlon = 86135.22228636884    # X轴初始位置
    ylat = 3282192.263342826    # Y轴初始位置
    vx0 = -1.0925598935349923    # X轴初始速度
    vy0 = -1.9232532244711842    # Y轴初始速度
    xa0 = 0    # X轴初始加速度
    ya0 = 0    # Y轴初始加速度
    x0 = np.array([[xlon], [ylat], [vx0], [vy0], [xa0], [ya0]])  # 初始状态向量，例如 [X,Y位置, X,Y速度]

    # 创建卡尔曼滤波器实例
    kf = KalmanFilter(F, B, H, Q, R, P, x0)

    for i in range(len(lon)):
        x, y = WGS84_to_plane(lon[i], lat[i])
        lon_plane.append(x)
        lat_plane.append(y)
        vx, vy = speed_transform(sog=sog[0], cog=cog[0])
        vx_plane.append(vx)
        vy_plane.append(vy)
    print(lon_plane, lat_plane)
    print(vx_plane, vy_plane)
    plt.figure()
    plt.scatter(lon_plane, lat_plane, color='blue', s=5)
    plt.plot(lon_plane, lat_plane, color='green')
    # plt.show()

    # 模拟观测值
    xsim = []   # x方向的预测值
    ysim = []   # y方向的预测值

    vx_ls = []  # 各个时刻X轴方向的速度值，用于三次样条插值，求X轴方向加速度
    vy_ls = []  # 各个时刻Y轴方向的速度值，用于三次样条插值，求Y轴方向加速度
    # 模拟过程
    for i in range(len(lon_plane)-1):
        # 预测步骤
        u = 0  # 假设控制输入为0.1，例如加速度
        x_pred, P_pred = kf.predict(u)
        print(f"Predicted state: {x_pred}")

        if i == 0:
            xsim.append(xlon)
            ysim.append(ylat)
            vx_ls.append(((i+1)*T, vx0))
            vy_ls.append(((i+1)*T, vy0))
        xsim.append(x_pred[0][0])
        ysim.append(x_pred[1][0])
        vx_ls.append(((i+2)*T, x_pred[2][0]))
        vy_ls.append(((i+2)*T, x_pred[3][0]))

        # 模拟观测值
        # z = np.dot(kf.H, x_pred) + np.random.randn() * np.sqrt(kf.R[0, 0])  # 观测噪声
        z = np.array([[x_pred[0][0]], [x_pred[1][0]], [x_pred[2][0]], [x_pred[3][0]]])
        # z = np.array([[lon_plane[i+1]], [lat_plane[i+1]], [vx_plane[i+1]], [vy_plane[i+1]]])

        ax = cubic_spline(vx_ls)
        ay = cubic_spline(vy_ls)

        # 更新步骤
        x_upd, P_upd = kf.update(z, ax, ay)
        print(f"Updated state: {x_upd}")


    print(xsim, ysim)
    print("vx_ls:", vx_ls)
    print("vy_ls:", vy_ls)
    plt.figure()
    plt.plot(xsim, ysim, color='red')
    plt.scatter(xsim, ysim, color='blue')
    # plt.plot(x, ypred, color='blue')
    plt.show()





