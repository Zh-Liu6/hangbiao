# 时间间隔设置为1，每秒模拟出轨迹进行预测
from cProfile import label
from datetime import *
from math import *

import plotly.graph_objs
from isort.profiles import plone
from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sphinx.ext.autosummary import periods_re


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

def cal_date_interval(base, new):
    base = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
    new = datetime.strptime(new, "%Y-%m-%d %H:%M:%S")
    interval = new - base
    return interval.total_seconds() + 1

def tra_predict(aisdata, ais_std):
    kf = KalmanFilter(dim_x=4, dim_z=4)
    dt0 = aisdata[0][0]
    x0 = aisdata[1][0]
    y0 = aisdata[2][0]
    vx0 = aisdata[3][0]
    vy0 = aisdata[4][0]
    kf.F = np.array([[1., dt0, 0., 0.],  # x   = x0 + dx*dt
                     [0., 1., 0., 0.],  # dx  = dx0
                     [0., 0., 1., dt0],  # y   = y0 + dy*dt
                     [0., 0., 0., 1.]])  # dy  = dy0
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    kf.x = np.array([[x0, vx0, y0, vy0]]).T
    kf.Q *= 0.1
    # 影响速度的计算
    kf.P *= 1
    last_t = 0
    # x, y = [x0], [y0]
    x, y = [], []
    cs_x = CubicSpline(aisdata[0], aisdata[3])  # 对X轴速度三次样条插值
    cs_y = CubicSpline(aisdata[0], aisdata[4])

    predict_time = 40 # 预测时间（s）
    # 前面的点做为训练，后面predict_time预测
    for i in range(1, len(aisdata[0])+predict_time):
        kf.R[0, 0] = ais_std ** 2
        kf.R[1, 1] = ais_std ** 2
        kf.R[2, 2] = ais_std ** 2
        kf.R[3, 3] = ais_std ** 2

        if i < len(aisdata[0]):
            dt = aisdata[0][i] - last_t
            last_t = aisdata[0][i]
            kf.F[0, 1] = dt
            kf.F[2, 3] = dt
            kf.predict()
            z = np.array([[aisdata[1][i], aisdata[3][i], aisdata[2][i], aisdata[4][i]]]).T  # 以实际数据（经纬度，航速航向）为观测值
            kf.update(z, R=kf.R*5)
        else:
            dt = 1
            now = last_t + i - len(aisdata[0])
            kf.F[0, 1] = dt
            kf.F[2, 3] = dt
            kf.predict()
            vx = cs_x(now).tolist()     # 三次样条插值得到速度
            vy = cs_y(now).tolist()
            z = np.array([kf.x[0][0], vx, kf.x[2][0], vy])    # 以预测值为观测值
            kf.update(z, R=kf.R*5)
            x.append(kf.x[0][0])
            y.append(kf.x[2][0])
    plt.plot(x, y, 'v', color='yellow')  # label='kalman predict'


    # 三次样条插值预测（分别预测X方向位置和Y方向位置）
    # cs_lon = CubicSpline(aisdata[0], aisdata[1])
    # cs_lat = CubicSpline(aisdata[0], aisdata[2])
    # time_pre = np.linspace(aisdata[0][-1], aisdata[0][-1]+predict_time, 100)
    # lon_pre = cs_lon(time_pre)
    # lat_pre = cs_lat(time_pre)
    # plt.plot(lon_pre, lat_pre, color='green', label='CubicSpline predict')


    # 三次样条插值预测（使用X方向位置预测Y方向位置）
    # 使用 sorted 函数对 x 进行排序，并获取排序后的索引
    sorted_indices = sorted(range(len(aisdata[1])), key=lambda k: x[k])
    sorted_x = [aisdata[1][i] for i in sorted_indices]
    sorted_y = [aisdata[2][i] for i in sorted_indices]
    cs_pre = CubicSpline(sorted_x, sorted_y)
    y_cs = cs_pre(x)
    plt.plot(x, y_cs, '*', color='green')   # label='CS X and Y'


    # 使用numpy进行多项式拟合
    # coefficients = np.polyfit(sorted_x, sorted_y, 2)  # deg是多项式的阶数,返回多项式的系数
    # fx = np.poly1d(coefficients)  # 拟合出的函数
    # y_fit = fx(x)  # 带入x轴坐标计算y轴坐标
    # plt.plot(x, y_fit, '^', color='c')

    # y_merge = []
    # for i in range(len(x)):
    #     a = y[i]*0.3 + y_cs[i]*0.7
    #     y_merge.append(a)
    # plt.plot(x, y_merge, '^', color='black')







if __name__ == '__main__':
    mmsi = 413762545
    csvpath = './data/' + str(mmsi) + 'zw.csv'
    df = pd.read_csv(csvpath, header=None, index_col=False, usecols=[1, 2, 3, 4, 6])  # time,lon,lat,cog,sog
    start = 0
    end = 30
    # start到end之间的数据为实际AIS数据，用于训练，训练完之后预测40s时间
    time_base = df[1][0]
    for i in range(5):  # 5个点预测
        ais_for_pre = df.iloc[start+i:end+i, :]
        ais_for_pre = np.array(ais_for_pre)
        timelist = []
        lonlist = []
        latlist = []
        vxlist = []
        vylist = []
        for j in range(len(ais_for_pre)):
            time = cal_date_interval(time_base, ais_for_pre[j][0])
            timelist.append(time)
            lon, lat = WGS84_to_plane(ais_for_pre[j][1], ais_for_pre[j][2])
            lonlist.append(lon)
            latlist.append(lat)
            vx, vy = speed_transform(ais_for_pre[j][4], ais_for_pre[j][3])
            vxlist.append(vx)
            vylist.append(vy)
        aisdata = [timelist, lonlist, latlist, vxlist, vylist]
        # 调用函数预测
        tra_predict(aisdata, 10)

    # 图上显示的实际AIS点
    ais_trajectory = df.iloc[end-5:end + 7, :]
    ais_trajectory = np.array(ais_trajectory)
    lon_true = []
    lat_true = []
    for i in range(len(ais_trajectory)):
        lon, lat = WGS84_to_plane(ais_trajectory[i][1], ais_trajectory[i][2])
        lon_true.append(lon)
        lat_true.append(lat)
    plt.plot(lon_true, lat_true, color='red', label='ais trajectory')
    plt.scatter(lon_true, lat_true, color='red', label='ais')
    # plt.legend(loc='best')
    plt.show()


