# 动态更新状态矩阵F中的时间
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
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt0 = aisdata[0][0]
    x0 = aisdata[0][1]
    y0 = aisdata[0][2]
    vx0 = aisdata[0][3]
    vy0 = aisdata[0][4]
    kf.F = np.array([[1., dt0, 0., 0.],  # x   = x0 + dx*dt
                     [0., 1., 0., 0.],  # dx  = dx0
                     [0., 0., 1., dt0],  # y   = y0 + dy*dt
                     [0., 0., 0., 1.]])  # dy  = dy0
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 0., 1., 0.]])
    kf.x = np.array([[x0, vx0, y0, vy0]]).T
    kf.Q *= 0.1
    # 影响速度的计算
    kf.P *= 100
    last_t = 0
    x, y = [x0], [y0]
    for i in range(1, len(aisdata)):
        kf.R[0, 0] = ais_std**2
        kf.R[1, 1] = ais_std**2
        dt = aisdata[i][0] - last_t
        last_t = aisdata[i][0]
        kf.F[0, 1] = dt
        kf.F[2, 3] = dt

        test_vx = kf.x[1][0]
        test_vy = kf.x[3][0]
        print(f"vx={test_vx}\tvy={test_vy}")
        kf.predict()

        z = np.array([[aisdata[i][1], aisdata[i][2]]]).T  # 以实际数据为观测值
        # z = np.array([kf.x[0], kf.x[2]])    # 以预测值为观测值


        kf.update(z, R=kf.R*5)
        x.append(kf.x[0][0])
        y.append(kf.x[2][0])


    plt.plot(x, y, color='blue', label='kalman predict')
    plt.scatter(x, y)
    plt.legend(loc='best')
    print(x, y)
    plt.show()




if __name__ == '__main__':
    df = pd.read_csv("./data/output.csv", header=None, index_col=False, usecols=[1, 2, 3, 4, 6])  # time,lon,lat,cog,sog
    df = df.iloc[:5, :]
    df = np.array(df)
    time_base = df[0][0]
    aisdata = []
    lon_true = []
    lat_true = []
    for i in range(len(df)):
        time = cal_date_interval(time_base, df[i][0])
        lon, lat = WGS84_to_plane(df[i][1], df[i][2])
        vx, vy = speed_transform(df[i][4], df[i][3])
        aisdata.append([time, lon, lat, vx, vy])
        lon_true.append(lon)
        lat_true.append(lat)
    plt.figure()
    print(lon_true, lat_true)
    plt.plot(lon_true, lat_true, color='red', label='ais')
    plt.scatter(lon_true, lat_true)
    tra_predict(aisdata, 10)




