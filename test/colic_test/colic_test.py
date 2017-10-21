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
import operator

#使用改进的随机梯度上升算法，通过训练样本产生回归系数；
#利用回归系数对测试样本进行分类，统计分类错误率
def colicTest():
    frTrain = open('horseColicTraining.txt') #打开训练样本集
    frTest = open('horseColicTest.txt') #打开测试样本集
    trainingSet = [] #存储所有样本的特征值的列表变量
    trainingLabels = [] #存储所有样本分类信息的列表变量
    #每个样本中有21个特征值，存储在变量trainingSet中；每个样本的分类信息存储在变量trainingLabels中
    for line in frTrain.readlines(): #从训练样本文件中逐个取出样本
        currLine = line.strip().split('\t') #去掉行首行末的空格，并用空格分隔，即将22个数值取出存到变量中
        lineArr = []
        for i in range(21): #将21个特征值从列表变量currLine中读取出来，储存到列表变量lineArr中
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr) #将当前样本20个特征值添加到列表trainingSet中
        trainingLabels.append(float(currLine[21])) #将当前样本分类信息添加到列表trainingLabels中
    #调用改进的随机梯度上升算法对训练样本进行500次迭代训练，产生回归系数
    trainWeights = Logistic.stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    #使用获取的回归系数，对测试样本进行分类，判断分类正确情况
    for line in frTest.readlines():
        numTestVec += 1.0 #统计测试样本个数
        currLine = line.strip().split('\t') #去掉行首行末的空格，并用空格分隔，即将22个数值取出存到变量中
        lineArr = []
        for i in range(21): #将21个特征值从列表变量currLine中读取出来，储存到列表变量lineArr中
            lineArr.append(float(currLine[i]))
        #判断当前测试样本的分类情况是否与真实分类一致；如果不一致，错误次数加1
        if int(Logistic.classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) #计算错误率
    print "the error rate of this test is : %f" % errorRate
    return errorRate #返回错误率

#多次测试
def multiTest():
    numTests = 10
    errorSum = 1.0
    for k in range(numTests): #统计10次测试的错误率，并累加
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % \
    (numTests, errorSum / float(numTests)) #打印出10次平均错误率

multiTest()
