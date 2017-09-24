# encoding: utf-8
'''
Created on Sep 16, 2017
创建决策树模块

@author: steve.wang
'''
import sys
from math import log
import operator

#获取训练样本dataSet的香农熵
def calcShannonEnt(dataSet):
        numEntries = len(dataSet) #获取样本集合中样本数
        labelCounts = {} #定义字典类型变量，存储各个分类中的样本个数
        for featVec in dataSet:
                #获取当前样本的分类信息，分类信息位于样本中最后一项，所以可以通过list的[-1]方式获取。
                currentLabel = featVec[-1]
                #labelCounts.keys()列举了字典变量中存储的所有关键字
                #判断当前样本的分类信息currentLabel是否存在于字典变量labelCounts中
                if currentLabel not in labelCounts.keys():
                        labelCounts[currentLabel] = 0 #将新的分类信息添加到字典变量labelCounts中
                labelCounts[currentLabel] += 1 #根据分类信息统计样本个数
        shannonEnt = 0.0
        #循环从类中依次求得所有类的熵之和：-p(xi)logP(xi)
        #p(xi)=该类所有样本/所有样本数
        for key in labelCounts:
                prob = float(labelCounts[key]) / numEntries
                shannonEnt -= prob * log(prob, 2)
        return shannonEnt

def createDataSet():
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataSet, labels

#从训练样本dataSet中查找所有特征axis与value一样的样本，并去除这些样本中的axis特征
def splitDataSet(dataSet, axis, value):
        retDataSet = [] #定义返回结果的列表变量
        for featVec in dataSet:
                if featVec[axis] == value: #查找样本中axis对应的特征的值等于变量value的样本
                        reduceFeatVec = featVec[: axis] #获取axis特征前面的特征
                        #a=[1,2,3], b=[4,5,6]
                        #a.append(b): [1, 2, 3, [4, 5, 6]]
                        #a.extend(b): [1, 2, 3, 4, 5, 6]
                        reduceFeatVec.extend(featVec[axis+1:]) #获取axis特征后面的特征
                        retDataSet.append(reduceFeatVec) #将样本中去除axis特征之后的样本添加到返回变量retDataSet
        return retDataSet

#返回样本集合dataSet中信息增益最大的特征所对应的id值
def chooseBestFeatureToSplit(dataSet):
        numFeatures = len(dataSet[0]) - 1 #获取样本中特征的数目，len(dataSet[0])第一个样本的长度，减去1，即减去类别标记
        baseEntropy = calcShannonEnt(dataSet) #获取整个训练样本的香农熵
        bestInfoGain = 0.0 #用于记录最佳特征的信息增益
        bestFeature = -1 #用于记录信息增益最大的特征
        for i in range(numFeatures): #按特征数依次循环
                featList = [example[i] for example in dataSet] #从训练样本dataSet中获取每个样本的第i个特征，存储到featList中
                #python的set和其他语言类似, 是一个无序不重复元素集, 基本功能包括关系测试和消除重复元素.
                #集合对象还支持union(联合), intersection(交), difference(差)和sysmmetric difference(对称差集)等数学运算.
                #sets 支持 x in set, len(set),和 for x in set。作为一个无序的集合，sets不记录元素位置或者插入点。
                #因此，sets不支持 indexing, slicing, 或其它类序列（sequence-like）的操作。
                #a = [3, 2, 3, 1, 2, 1]
                #set(a) : [1, 2, 3]
                uniqueVals = set(featList) #去除包含特征i的列表中重复项，并从小到大进行排序，生成集合变量uniqueVals
                newEntropy = 0.0
                #获取第i个特征所对应的样本集合的香农熵
                for value in uniqueVals: #按照第i个特征的不同取值进行循环
                        subDataSet = splitDataSet(dataSet, i, value) #获取训练样本中第i个特征值与value变量相等的样本集合
                        prob = len(subDataSet) / float(len(dataSet)) #计算subDataSet在训练集合中所占比例
                        #获取样本集subDataSet的香农熵，将不同特征取值的香农熵按其所占比例累加起来
                        newEntropy += prob * calcShannonEnt(subDataSet)
                infoGain = baseEntropy - newEntropy #计算整个样本的香农熵与当前特征所对应样本集合的香农熵之差，用于判断当前特征的信息增益大小
                if infoGain > bestInfoGain: #如果第i 个特征的信息增益大于之前最大的信息增益，则更新最佳特征的相关信息
                        bestInfoGain = infoGain #更新最佳信息增益
                        bestFeature = i #更新最佳特征id
        return bestFeature #返回最佳特征所对应的id

#从存储分类信息列表classList中获取出现次数最多的分类名称
def majorityCnt(classList):
        classCount = {} #定义存储各分类样本数目的字典变量
        for vote in classList:
                if vote not in classCount.keys(): #判断新的分类信息是否已经存在字典变量classCount中
                        classCount[vote] = 0 #如果不存在，则将该分类保存在字典变量classCount中
                classCount[vote] += 1 #相应分类的样本数目加1
        #将字典变量classCount按照其存储分类样本数从大到小排序
        sortedClassCount = sorted(classCount.iteritems(),
                                  key = operator.itemgetter(1), reverse = True)
        return sortedClassCount[0][0] #返回出现次数最多的分类名称，即字典变量classCount对应的key

#dataSet为所有训练样本
#labels为训练样本所有特征对应的分类信息
#产生dataSet所对应的树结构信息，以字典形式存储并返回
def createTree(dataSet, labels):
        #获取训练样本dataSet中所有的分类标签信息，存储在列表变量classList中
        classList = [example[-1] for example in dataSet]
        #classList[0]表示第一个分类信息，在所有分类信息中查找与classList[0]一样的样本数目
        #如果该数目等于列表变量classList的元素数目，则表示所有分类都相同，返回这个分类信息
        if classList.count(classList[0]) == len(classList):
                return classList[0]
        if len(dataSet[0]) == 1: #如果当前训练样本中只剩下分类信息了，则返回样本集中数量最多的分类信息。
                return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplit(dataSet) #获取训练样本dataSet中信息增益最大的特征ID
        bestFeatLabel = labels[bestFeat] #获取最佳特征ID对应的分类信息
        myTree = {bestFeatLabel:{}} #定义字典变量myTree，用于存储当前训练集中以最佳特征为根节点的树
        del(labels[bestFeat]) #从分类信息中删除最佳特征所对应的分类信息        
        #或许训练样本dataSet中最佳特征所对应的信息，并存储在列表变量featValues中
        featValues = [example[bestFeat] for example in dataSet]
        print featValues
        uniqueVals = set(featValues) #去除最佳特征列表中重复项，并从小到大进行排序，生成集合变量uniqueVals
        for value in uniqueVals: #循环处理最佳特征的所有属性信息
                subLabels = labels[:] #将label中所有数据复制到subLabels变量中，以防止在后面改变label中的数据
                #递归调用createTree函数，产生从当前训练样本中去除最佳特征后的训练集所对应的树数据
                #该变量myTree以树的形式依次保存各个分支的信息，直到叶子结点
                myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)                
        return myTree #返回当前训练样本集合所产生的树结构信息

#inputTree为以字典形式构建的决策树
#featLabels存储着所有分类项
#testVec为测试样本
#该函数在决策树inputTree查找测试样本testVec的分类信息
def classify0(inputTree, featLabels, testVec):
        firstStr = inputTree.keys()[0] #获取决策树的根节点，也为特征值
        secondDict = inputTree[firstStr] #获取决策树中根节点下面的各个子树
        #index() 函数用于从列表中找出某个值第一个匹配项的索引位置
        featIndex = featLabels.index(firstStr) #找出根节点对应的特征ID
        #字典(Dictionary) keys() 函数以列表返回一个字典所有的键
        for key in secondDict.keys(): #循环遍历各个子树的键值，即特征值                
                if testVec[featIndex] == key: #如果测试样本的特征与当前特征一样，则进一步处理
                        #如果仍为子树，即类型为dict，则递归继续处理
                        if type(secondDict[key]).__name__ == 'dict':
                                classLabel = classify(secondDict[key], featLabels, testVec)
                        else:
                                classLabel = secondDict[key] #如果是叶子节点，则返回对应类别
        return classLabel #返回查找到的类别信息

#inputTree为以字典形式构建的决策树
#featLabels存储着所有分类项
#testVec为测试样本
#该函数在决策树inputTree查找测试样本testVec的分类信息
def classify(inputTree, featLabels, testVec):
        firstStr = inputTree.keys()[0] #获取决策树的根节点，也为特征值
        secondDict = inputTree[firstStr] #获取决策树中根节点下面的各个子树
        #index() 函数用于从列表中找出某个值第一个匹配项的索引位置        
        featIndex = featLabels.index(firstStr) #找出根节点对应的特征ID
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        #isinstance（object，type）
        #来判断一个对象是否是一个已知的类型。
        #其第一个参数（object）为对象，第二个参数（type）为类型名(int...)或类型名的一个列表((int,list,float)是一个列表)。
        #其返回值为布尔型（True or flase）。
        #若对象的类型与参数二的类型相同则返回True。若参数二为一个元组，则若对象类型与元组中类型名之一相同即返回True。
        if isinstance(valueOfFeat, dict):
                classLabel = classify(valueOfFeat, featLabels, testVec)
        else:
                classLabel = valueOfFeat #如果是叶子节点，则返回对应类别        
        return classLabel #返回查找到的类别信息
