import numpy as np
from pyproj import CRS, Transformer
from math import *
from matplotlib import pyplot as plt

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
        return self.x, self.P

    def update(self, z):
        """
        更新步骤
        :param z: 观测值
        :return: 更新的状态和协方差
        """
        y = z - np.dot(self.H, self.x)  # 残差
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R  # 残差协方差
        sni = np.linalg.inv(S)  # 求S的逆矩阵
        # K = np.dot(self.P, self.H.T) / S  # 卡尔曼增益
        K = np.dot(self.P, self.H.T).dot(sni)

        self.x = self.x + np.dot(K, y)  # 更新状态
        self.P = self.P - np.dot(K, self.H).dot(self.P)  # 更新协方差
        return self.x, self.P


# 在这个示例中，我们定义了一个一维运动对象的模型，其中状态向量 x 包括位置和速度。F 矩阵描述了状态如何随时间演变，B 矩阵表示控制输入（例如加速度）如何影响状态，
# H 矩阵定义了观测值（位置）是如何从状态向量中获得的。Q 和 R 分别是过程噪声和观测噪声的协方差矩阵，它们量化了预测和观测中的不确定性。初始状态 x0 和初始协方差 P 定义了滤波器开始时对状态的估计和不确定性。

if __name__ == '__main__':

    # 系统参数
    T = 1 # 采样时间间隔
    F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])  # 状态转移矩阵，例如 [位置, 速度]
    B = np.array([[0.5*T*T], [0.5*T*T], [T], [T]])  # 控制输入矩阵，例如 加速度影响  目前状态转移方程为x=Fx+Bu,X轴Y轴加速度一样, 加速度不一样：x=Fx+B1*u1+B2*u2
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 观测矩阵，观测位置
    Q = np.array([[0.1, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # 过程噪声协方差矩阵
    R = np.array([[0.1, 0], [0, 0.1]])  # 观测噪声协方差矩阵

    # 初始参数
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # 初始状态协方差矩阵
    x0 = np.array([[0], [0], [1], [2]])  # 初始状态向量，例如 [位置, 速度]

    # 创建卡尔曼滤波器实例
    kf = KalmanFilter(F, B, H, Q, R, P, x0)

    # 模拟观测值
    xsim = []
    ysim = []
    # 模拟过程
    for i in range(6):
        # 预测步骤
        u = 1  # 假设控制输入为0.1，例如加速度
        x_pred, P_pred = kf.predict(u)
        print(f"Predicted state: {x_pred}")

        xsim.append(x_pred[0][0])
        ysim.append(x_pred[1][0])

        # 模拟观测值
        # z = np.dot(kf.H, x_pred) + np.random.randn() * np.sqrt(kf.R[0, 0])  # 观测噪声
        z = np.array([[x_pred[0][0]], [x_pred[1][0]]])

        # 更新步骤
        x_upd, P_upd = kf.update(z)
        print(f"Updated state: {x_upd}")


    print(xsim, ysim)

    plt.figure()
    plt.plot(xsim, ysim, color='red')
    # plt.plot(x, ypred, color='blue')
    plt.show()

    # x, y = WGS84_to_plane(113.756, 34.26)
    # print(x, y)
    #
    # vx, vy = speed_transform(120, 30)
    # print(vx, vy)


# [1.5, 4.0, 7.5, 12.0, 17.5, 24.0] [3.0, 8.0, 15.0, 24.0, 35.0, 48.0]