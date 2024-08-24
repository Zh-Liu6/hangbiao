import numpy as np

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
        K = np.dot(self.P, self.H.T) / S  # 卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态
        self.P = self.P - np.dot(K, self.H).dot(self.P)  # 更新协方差
        return self.x, self.P

# 系统参数
F = np.array([[1, 1], [0, 1]])  # 状态转移矩阵，例如 [位置, 速度]
B = np.array([[0.5], [0.5]])    # 控制输入矩阵，例如 加速度影响
H = np.array([[1, 0]])          # 观测矩阵，观测位置
Q = np.array([[0.1, 0], [0, 0.01]])  # 过程噪声协方差矩阵
R = np.array([[1]])              # 观测噪声协方差矩阵

# 初始参数
P = np.array([[1, 0], [0, 1]])  # 初始状态协方差矩阵
x0 = np.array([[0], [0]])        # 初始状态向量，例如 [位置, 速度]

# 创建卡尔曼滤波器实例
kf = KalmanFilter(F, B, H, Q, R, P, x0)

# 模拟过程
for _ in range(5):
    # 预测步骤
    u = 0.1  # 假设控制输入为0.1，例如加速度
    x_pred, P_pred = kf.predict(u)
    print(f"Predicted state: {x_pred}")

    # 模拟观测值
    z = np.dot(kf.H, x_pred) + np.random.randn() * np.sqrt(kf.R[0, 0])  # 观测噪声
    # 更新步骤
    x_upd, P_upd = kf.update(z)
    print(f"Updated state: {x_upd}")

# 在这个示例中，我们定义了一个一维运动对象的模型，其中状态向量 x 包括位置和速度。F 矩阵描述了状态如何随时间演变，B 矩阵表示控制输入（例如加速度）如何影响状态，
# H 矩阵定义了观测值（位置）是如何从状态向量中获得的。Q 和 R 分别是过程噪声和观测噪声的协方差矩阵，它们量化了预测和观测中的不确定性。初始状态 x0 和初始协方差 P 定义了滤波器开始时对状态的估计和不确定性。


