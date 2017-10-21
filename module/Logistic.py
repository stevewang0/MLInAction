# encoding: utf-8
'''
Created on Oct 4, 2017
创建Logistic模块

@author: steve.wang
'''

from numpy import *

#sigmoid函数的实现
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#dataMatIn：训练样本矩阵，每个样本包含3个特征值
#classLabels：训练样本对应的分类矩阵
#该函数利用梯度上升算法，产生经过多次调整后的回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #调用mat()函数将数组转化为矩阵
    labelMat = mat(classLabels).transpose() #调用transpose()函数对矩阵进行转置
    m, n = shape(dataMatrix) #获取矩阵或者数组的维数
    alpha = 0.001 #向目标移动的步长
    maxCycles = 500 #迭代次数
    weights = ones((n, 1)) #产生n*1的矩阵，初始化为全1；该变量存储回归系数
    #循环500次，每次按照当前的回归系数，计算每个样本的预测类别；计算出其与真实类别的差值，进而调整回归系数，然后再循环处理。
    for k in range(maxCycles):
        #dataMatrix * weights为矩阵乘法，dataMatrix为mxn的矩阵，weights为nx1的矩阵，它们乘积为mx1的矩阵
        h = sigmoid(dataMatrix * weights) #针对dataMatrix * weights的乘积结果，获取各个元素对应的sigmoid值，h为mx1的矩阵
        error = (labelMat - h) #获取各个样本预测类别与真实类别信息数据的差值
        #根据差值error的方向调整回归系数，如果error为负值 ，则减小系数；反之，增大系数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#dataMatIn：训练样本矩阵，每个样本包含3个特征值
#classLabels：训练样本对应的分类矩阵
#该函数利用随机梯度上升算法，产生按照样本数进行多次调整后的回归系数
def stocGradAscent0(dataMatrix, claaLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01 #向目标移动的步长
    weights = ones(n) #创建有n个元素的回归系数数组，并初始化为1
    for i in range(m): #按照样本数目进行循环迭代
        h = sigmoid(sum(dataMatrix[i] * weights)) #求得第i个样本的特征值与回归系数乘积之和，然后调用函数sigmoid获得分类结果
        error = classLabels[i] - h #计算计算的分类结果与真实结果差值
        weights = weights + alpha * error * dataMatrix[i] #对回归系数进行调整
    return weights

#dataMatIn：训练样本矩阵，每个样本包含3个特征值
#classLabels：训练样本对应的分类矩阵
#numIter：迭代次数，默认值为150
#该函数利用随机梯度上升算法，多次迭代，产生多次调整后的回归系数
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n) #创建有n个元素的回归系数数组，并初始化为1
    for j in range(numIter): #迭代处理，默认迭代150次
        #range函数用来创建序列，rang(5)结果为[0,1,2,3,4]
        dataIndex = range(m) 
        for i in range(m): #每次迭代，需要更新m次系数，m等于样本次数
            alpha = 4 / (1.0 + j + i) + 0.01 #每次迭代时系数调整的步长
            #numpy.random.uniform(low,high,size)：
            #从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
            #从样本中随机选择一个样本对应的index，作为此次迭代的样本id
            randIndex = int(random.uniform(0, len(dataIndex)))
            #求得第randIndex个样本的特征值与回归系数乘积之和，然后调用函数sigmoid获得分类结果
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h #计算计算的分类结果与真实结果差值
            weights = weights + alpha * error * dataMatrix[randIndex] #对回归系数进行调整
            del(dataIndex[randIndex]) #从样本ID集中删除此次迭代所用样本ID
    return weights

#inX：测试样本
#weights：训练好的系数
#获得测试样本的分类情况
def classifyVector(inX, weights):
    #求得测试样本的特征值与回归系数乘积之和，然后调用函数sigmoid获得分类结果
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: #如果分类结果大于0.5，则认为类别1；否则分为类别0
        return 1.0
    else:
        return 0.0
