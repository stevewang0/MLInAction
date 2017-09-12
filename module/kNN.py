# encoding: utf-8
'''
Created on Sep 3, 2017
kNN: k Nearest Neighbors

Input:			inX: 1行M列的向量，为测试样本，用来跟输入变量dataSet中的数据进行比较(1xM)
						dataSet: 已知的N行M列的数组，为训练样本集，一行为一个训练样本，即N个训练样本，每个训练样本M个元素(NxM)
            labels: 1行N列的向量，存储各个训练样本对应分类的标签信息 (1xN)
            k: 用来比较的近邻元素的个数，选择奇数，且小于20
            
Output:     测试样本最接近的分类标签

@author: steve.wang
'''

from numpy import *
import operator

def knnClassify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0] #获取训练样本集dataSet的个数，即矩阵dataSet的行数N
  #tile(inX, (dataSetSize, 1))是建立一个N行1列的数组，但其元素是一个1行M列的向量，最后返回一个N行M列的数组
  #N行M列的数组中每一行都是输入的测试样本，它与训练样本集相减，则得到NxM的数组值之差
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1) #axis=1是对存储NxM个元素对应的平方差的数组，将每一行的值累加起来，返回一个Nx1的数组
	distances = sqDistances**0.5 #求得测试样本与各个训练样本的欧式距离
    
  #对distances中N个元素进行从小到大排序，之后返回一个N个元素的一维数组，存储distances排序后各个元素在原来数组中的index
  #eg. distances=[2,1,3,0], argsort的返回值为[3，1，0，2]
	sortedDistIndicies = distances.argsort()
    
	classCount={} #定义一个空的字典变量
	for i in range(k): #i的取值为[0，k - 1]
		voteLabel = labels[sortedDistIndicies[i]] #返回测试样本与训练样本欧式距离第i小的训练样本所对应的类的标签
		#classCount.get(voteLabel, 0)获取classCount中voteLabel为index的元素的值，找不到则返回0
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        
  #classCount.iteritems()返回字典classCount中所有项，按照迭代器的方式
  #operator.itemgetter()用于获取对象的哪些维的数据，参数为一些序号；此处参数为1，即按照字典变量classCount各项的第二个元素进行排序
  #reverse = True表示按照逆序排序，即从大到小的次序；False表示按照从小到大的次序排序
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    
  #返回k个近邻元素中所对应类最多那个类的标签，即测试样本所属那个类
	return sortedClassCount[0][0]
