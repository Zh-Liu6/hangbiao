from docutils.nodes import header
from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
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

from sympy import symbols, diff

x = symbols('x')  # 定义符号
f = x**2  # 示例函数

# 求一阶导数
f_prime = diff(f, x)
print(f"一阶导数为{f_prime}")

x_point = 2
value = f_prime.subs(x, x_point)
print(f"在x_point处的导数为{value}")

# 求二阶导数
f_double_prime = diff(f, x, x)
print(f"二阶导数为{f_double_prime}")

df = pd.read_csv("data/aistra.csv", header=None, index_col=False)  # 参数usecols=[0, 1]，表示读取CSV的第一列和第二列

lon = df.iloc[:, 2]
lat = df.iloc[:, 3]
cog = df.iloc[:, 4]
sog = df.iloc[:, 6]

plt.figure()
plt.scatter(lon, lat, color='blue', s=5)
plt.plot(lon, lat, color='green')
# 设置图表标题和坐标轴标签
plt.title('Plot of X vs Y')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
# plt.grid(True)    # 显示网格
plt.show()



# ais = pd.read_csv('./data/pre_data.csv', header=None, index_col=False)
# # 筛选MMSI为413762545的行
# ais = ais[ais.iloc[:, 0]==413762545]
# # 将DataFrame保存到CSV文件，不包含列名和索引
# ais.to_csv('./data/output.csv', index=False, header=False)

ais = pd.read_csv('./data/output.csv', header=None, index_col=False)

plt.figure()
plt.scatter(ais[2], ais[3], color='blue', s=5)
plt.plot(ais[2], ais[3], color='green')
# 设置图表标题和坐标轴标签
plt.title('Plot of X vs Y')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
# plt.grid(True)    # 显示网格
plt.show()

print("hello")