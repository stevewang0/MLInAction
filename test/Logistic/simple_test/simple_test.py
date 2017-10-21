# encoding: utf-8
'''
Created on Oct 13, 2017
测试Logistic模块

@author: steve.wang
'''
import sys
sys.path.append(r'../../../module') #将algo模块所在路径添加进来
import Logistic
from numpy import *

#从训练样本文件中读取样本信息，产生训练样本矩阵及分类矩阵
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt') #打开样本文件
    for line in fr.readlines(): #从样本文件中读取每行数据，即一个样本
        #strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
        #split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串。
        lineArr = line.strip().split() #分离一个样本的两个特征及其分类信息
        #将每个样本的两个特征作为训练矩阵的后面两个特征，第一特征固定设置为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2])) #存储训练样本的类型
    return dataMat, labelMat #返回训练样本的矩阵和类别矩阵

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights =  wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dataArr, labelMat = loadDataSet()
weight = Logistic.gradAscent(dataArr, labelMat)
print weight
plotBestFit(weight)
