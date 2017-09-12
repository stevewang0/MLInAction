# encoding: utf-8
'''
Created on Sep 3, 2017
创建简单的二维数组，测试kNN模块的分类函数。

@author: steve.wang
'''
import sys
sys.path.append(r'../../../module') #将kNN模块所在路径添加进来
import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group = array([[1.1, 1.1], [1.0, 1.2], [0.2, 0.4], [0.9, 0.1]]) #创建4x2的数组作为训练样本
	labels = ['A', 'A', 'B', 'B'] #4个训练样本所属类的标记
	
	return group, labels

group, labels = createDataSet() #创建训练样本和对应标记
realIndex = kNN.knnClassify([0.2, 0], group, labels, 3) #检测测试样本[0.2, 0]属于哪个类
print realIndex

#Matplotlib 里的常用类的包含关系为 Figure -> Axes -> (Line2D, Text, etc.)
#一个Figure对象可以包含多个子图(Axes)，在matplotlib中用Axes对象表示一个绘图区域，可以理解为子图。
fig = plt.figure() #创建图表fig
ax = fig.add_subplot(1, 1, 1) #在图表fig中创建一个子图ax
#绘制散列图，前两个参数表示x轴和y轴所要显示的数据，s表示符号大小，c表示颜色，marker表示符号类型
#ax.scatter(group[:, 0], group[:, 1], s = 50, c = 'r', marker = 'o')
label = list(ones(2))+list(2*ones(2)) #定义4个元素的数组
#使用数组label中的值来改变s和c的值
ax.scatter(group[:, 0], group[:, 1], 15.0 * array(label), 15.0 * array(label), marker = 'o')
#设置坐标系x轴和y轴的上下限
ax.axis([0, 2, 0, 2])
ax.set_title('kNN_4_simple_group') #设置子图的的标题
plt.xlabel('x_value')
plt.ylabel('y_value')
plt.savefig("kNN_4_simple_group.pdf")
plt.show()
