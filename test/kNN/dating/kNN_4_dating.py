# encoding: utf-8
'''
Created on Sep 4, 2017
使用约会网站的数据，对这些数据进行分类处理。

@author: steve.wang
'''
import sys
sys.path.append(r'../../../module') #将kNN模块所在路径添加进来
import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#从训练样本文件中读取样本信息，并将分类信息与其他数据分别存储在不同的数组中
def file2matrix(filename):
    fd = open(filename)
    numberOfLines = len(fd.readlines()) #获取文件中样本的数目
    #创建numberOfLinesx3的二维数组，并初始化为0；存储从文件中读取的样本信息的前三列
    returnMat = zeros((numberOfLines, 3)) 
    classLabelVector = [] #定义存储样本类别的列表
    fd = open(filename) #重新打开文件，因为前面为了获取样本数目从文件中读取了所有样本，文件指针不在文件最前面
    index = 0
    for line in fd.readlines(): #从文件中循环读取每一行数据进行处理
        line = line.strip() #去掉这一行前面和后面的空格
        listFromLine = line.split('\t') #将数据按空格分割，本示例中将所读行数据分为4个独立数据
        returnMat[index, :] = listFromLine[0 : 3] #将前3个数据存储到returnMat数组中
        classLabelVector.append(int(listFromLine[-1])) #将最后一个数据，即第4个数据，转换成int类型后存储到classLabelVector最后面
        index += 1

    return returnMat, classLabelVector #returnMat存储样本前3列数据，classLabelVector存储样本对应的分类信息

#对输入数组进行归一化数值处理，也叫特征缩放，用于将特征缩放到同一个范围内
#本例的缩放公式为： newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet): #输入数组dataSet为Nx3的二维数组
    minVals = dataSet.min(0) #获取数组中每一列的最小值，minVals为1x3的数组；max(0)获取每一行的最小值
    maxVals = dataSet.max(0) #获取数组中每一列的最大值，maxVals为1x3的数组；max(1)获取每一行的最大值
    ranges = maxVals - minVals #获取特征范围差，ranges也是1x3的数组
    normDataSet = zeros(shape(dataSet)) #创建与dataSet的维数、类型完全一样的数组，并初始化为0，用于存储归一化后的结果
    m = dataSet.shape[0] #获取输入数组的行数
    #tile(minVals, (m, 1))创建mx1的数组，数组元素为1x3的最小值数组，其返回值为mx3的数组
    #从原始数组的各个元素中，减去对应列的最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    #tile(ranges, (m, 1))创建mx1的数组，数组元素为1x3的特征范围差，其返回值为mx3的数组
    #原始数组的各个元素除以对应列的特征范围差，完成归一化
    normDataSet = normDataSet / tile(ranges, (m, 1))
    
    return normDataSet, ranges, minVals #返回归一化后Nx3的数组、1x3的特征范围差数组和1x3的每列最小值数组

#根据训练样本datingTestSet2.txt中数据对kNN分类算法进行测试
def datingClassTest():
    hoRatio = 0.50 #用于分割样本，将文件中获取的样本前面一半作为测试样例，后面一半作为训练样例
    #从样本文件datingTestSet2.txt中读取所有样例数据及分类信息
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #将样本数据进行归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)    
    m = normMat.shape[0] #获取归一化后二维数组的行数，即所有样本的数目
    numTestVecs = int(m * hoRatio) #获取测试样本的数目，为所有样本的一半
    errorCount = 0.0 #记录分类错误的次数
    for i in range(numTestVecs): #依次循环，从前一半样本中获得每一个样本，跟后面一半样本进行比对，寻找最近邻样本
        classifierResult = kNN.knnClassify(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        #print "the classifier case back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): #如果分类结果与从文件中读取的值不一致，则判为分类错误
            errorCount += 1.0
            print "the classifier case back with: %d, the real answer is: %d, index: %d" % (classifierResult, datingLabels[i], i)
    print "the total error rate is: %f" % (errorCount / float(numTestVecs)) #打印出错误率
    print errorCount #返回错误次数

#用户输入某人信息判断其分类
def datingClassifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses'] #定义三种分类的列表变量
    #input需要按照输入信息类型输入输出，字符串需要用''引起；数字直接输入
    #raw_input的所有输入都会当作字符串，如果是数字需要显示转换为相应类型
    percentTats = float(raw_input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #获取训练样本中的数据
    normMat, ranges, minVals = autoNorm(datingDataMat) #对样本信息进行数值归一化
    inArr = array([ffMiles, percentTats, iceCream]) #使用输入信息定义数组变量
    #(inArr - minVals) / ranges：将输入用户信息归一化，然后再与归一化后的训练样本进行kNN分类
    classifierResult = kNN.knnClassify((inArr - \
                                        minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", \
          resultList[classifierResult - 1] #返回对应分类的字符串
    

datingClassTest()
#Matplotlib 里的常用类的包含关系为 Figure -> Axes -> (Line2D, Text, etc.)
#一个Figure对象可以包含多个子图(Axes)，在matplotlib中用Axes对象表示一个绘图区域，可以理解为子图。
fig = plt.figure() #创建图表fig
ax = fig.add_subplot(1, 1, 1) #在图表fig中创建一个子图ax
#绘制散列图，前两个参数表示x轴和y轴所要显示的数据，s表示符号大小，c表示颜色
#使用数组label中的值来改变s和c的值
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
ax.scatter(datingDataMat[: , 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
#设置坐标系x轴和y轴的上下限
ax.axis([0, 25, 0, 2])
ax.set_title('kNN_4_dating') #设置子图的的标题
plt.xlabel('time for video game')
plt.ylabel('weight of ice cream')
plt.savefig("kNN_4_dating.pdf")
plt.show()

datingClassifyPerson()

