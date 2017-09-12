# encoding: utf-8
'''
Created on Sep 7, 2017
手写数字识别系统

@author: steve.wang
'''
import sys
sys.path.append(r'../../../module') #将kNN模块所在路径添加进来
import kNN
from numpy import *
from os import listdir

#将文件中读取的数字字符转换为整型数字，存储在一维数组中
def img2vector(filename):
    returnVect = zeros((1, 1024)) #创建1x1024的一维数组，并初始化为0
    #打开文件从中读取32行数据，并将每一行的32个byte转化为整型数字
    fd = open(filename) 
    for i in range(32):
        lineStr = fd.readline() #读取一行数据
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j]) #每个byte转换为整型数字
    return returnVect

def handwritingClassTest():
    hwLabels = [] #定义存储训练样本类别标记的列表变量
    #listdir(): 返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    #目录trainingDigits下面存放着所有的训练样本文件，listdir获取这些文件名将其存储在列表变量trainingFileList中
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList) #获取文件列表中的文件个数
    trainingMat = zeros((m, 1024)) #创建mx1024的二维数组且初始化为0，存储训练样本的内容；每个样本1024个byte
    for i in range(m): #逐个从样本文件中读取内容转换成整型存储在二维数组trainingMat中
        fileNameStr = trainingFileList[i] #获取第i个样本的文件名0_3.txt，表示数字0的第3个样本文件
        fileStr = fileNameStr.split('.')[0] #获取样本文件名称0_3
        classNumStr = int(fileStr.split('_')[0]) #获取该样本对应的数字类别，即0
        hwLabels.append(classNumStr) #将样本的数字类别存储在列表变量hwLabels中
        #将文件trainingDigits/0_3.txt读取出来，并将其内容转换为整型数字存储在二维数组trainingMat第i个元素中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits') #测试样本存放在testDigits目录下，获取测试样本的文件名列表
    errorCount = 0.0
    mTest = len(testFileList) #获取测试样本的个数
    for i in range(mTest): #对测试样本，逐个进行检测
        fileNameStr = testFileList[i] #获取第i个测试样本，如0_1.txt
        fileStr = fileNameStr.split('.')[0] #获取文件名0_1
        classNumStr = int(fileStr.split('_')[0]) #获取样本的数字类别0
        #将文件testDigits/0_1.txt读取出来，并将其内容转换为整型数字存储在1x1024的数组vectorUnderTest中
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #将vectorUnderTest与之前的训练样本逐一进行比较，获取最近邻的3个样本，
        #返回这3个样本中类别最多的那个类别，作为该测试样本的类别
        classifierResult = kNN.knnClassify(vectorUnderTest, \
                                           trainingMat, hwLabels, 3)
        #print "the classifier came back with: %d, the real answer is: %d" \
        #      % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): #如果返回类别与真实类别不一致，则增加错误数量
            errorCount += 1.0
            print "the classifier came back with: %d, the real answer is: %d" \
                  % (classifierResult, classNumStr)
    print "\nthe total number of errors is: %d" % errorCount #打印出错误数量
    print "\nthe total error rate is: %f" % (errorCount / float(mTest)) #打印出错误率
        
handwritingClassTest()
