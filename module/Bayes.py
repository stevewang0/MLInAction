# encoding: utf-8
'''
Created on Sep 26, 2017
创建朴素贝叶斯模块

@author: steve.wang
'''
from numpy import *
import re

#dataSet：文档样本集
#创建dataSet包含的所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([]) #创建一个空集合变量
    for document in dataSet:
        #set(document)获得document中所有不重复的词
        #vocabSet | set(document)求集合的并集，返回不重复的词集合
        #在数学符号表示上，按位或操作与集合求并操作使用相同记号"|"
        vocabSet = vocabSet | set(document)
    return list(vocabSet) #将集合转换为列表并返回

#vocabList: 词汇表
#inputSet: 测试样本文档
#获取测试样本中单词在词汇表中是否出现的列表
def setOfWords2Vec(vocabList, inputSet):
    #定义一个列表变量，其长度与vocabList一样，其内容初始化为0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList: #判断测试文档中的单词是否存在于词汇表
            #index() 函数用于从列表中找出某个值第一个匹配项的索引位置
            #寻找当前单词在词汇表vocabList中的位置，在列表returnVec对应位置写1，表示该单词存在于词汇表
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#vocabList: 词汇表
#inputSet: 测试样本文档
#获取表示测试样本中单词在词汇表中出现次数的列表
def bagOfWords2Vec(vocabList, inputSet):
    #定义一个列表变量，其长度与vocabList一样，其内容初始化为0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList: #判断测试文档中的单词是否存在于词汇表
            #index() 函数用于从列表中找出某个值第一个匹配项的索引位置
            #寻找当前单词在词汇表vocabList中的位置，在列表returnVec对应位置累加1，记录该单词在词汇表中出现的次数
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#trainMatrix: 存储每个文档样本在词汇表中各个词汇出现情况的集合，与样本数目相同
#这是一个二维数组，第一维对应各个文档样本id，第二维对应该样本中词汇在词汇表中出现情况
#trainCategory: 存储每个文档样本所属类别的标签，即分类信息
#获取侮辱性和非侮辱性样本中各个词汇出现的概率，及样本集的侮辱性概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #获取文档样本个数
    numWords = len(trainMatrix[0]) #获取词汇表的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs) #获取所有样本的分类概率
    #利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
    #即计算p(w0|1)p(w1|1)p(w2|1)。如果其中一个概率值为0，那么最后的乘积也为0。
    #为降低这种影响，将所有词的出现数初始化为1，并将分母初始化为2。
    #p0Num = zeros(numWords) #创建有numWords个元素的数组，且每个元素初始化为0
    #p1Num = zeros(numWords) #创建有numWords个元素的数组，且每个元素初始化为0
    #p0Denom = 0.0 #存储所有样本中非侮辱性词汇出现次数总和
    #p1Denom = 0.0 #存储所有样本中侮辱性词汇出现次数总和
    p0Num = ones(numWords) #创建有numWords个元素的数组，且每个元素初始化为1
    p1Num = ones(numWords) #创建有numWords个元素的数组，且每个元素初始化为1
    p0Denom = 2.0 #存储所有样本中非侮辱性词汇出现次数总和
    p1Denom = 2.0 #存储所有样本中侮辱性词汇出现次数总和    
    for i in range(numTrainDocs): #逐个样本进行循环处理
        if trainCategory[i] == 1: #如果当前样本的分类结果为1，即abusive，则对p1xx进行调整
            #p1Num和trainMatrix[i]为含相同数目元素的list，此处的+是对应位置上元素的相加
            p1Num += trainMatrix[i]
            #sum(trainMatrix[i])求得第i个样本中所有侮辱性词汇出现的个数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #当计算乘积p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时，由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。
    #一种解决办法是对乘积取自然对数。在代数中有ln(a*b)=ln(a)*ln(b)，于是通过求对数可以避免下溢出或者浮点数舍入导致的错误。
    #同时，采用自然对数进行处理不会有任何损失。            
    #对list变量p1Num中每个元素都除以p1Denom，求得词汇表中每个词汇在所有侮辱性样本中出现的概率
    #p1Vect = p1Num / p1Denom
    p1Vect = log(p1Num / p1Denom)
    #对list变量p0Num中每个元素都除以p0Denom，求得词汇表中每个词汇在所有非侮辱性样本中出现的概率
    #p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

#vec2Classify：测试样本中单词在词汇表中出现情况信息
#p0Vec：非侮辱性词汇出现的概率
#p1Vec: 侮辱性词汇出现的概率
#pClass1: 训练样本中侮辱性样本的概率
#判断输入样本信息是否为侮辱性样本
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #sum(vec2Classify * p1Vec): 求两个向量对应元素乘积之和，计算测试样本中侮辱性词汇的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0: #比较侮辱性概率和非侮辱性概率，p1大，则为侮辱性样本；反之，为非侮辱性样本
        return 1
    else:
        return 0

#bigString：英文句子
#解析英文句子，将其中长度大于2单词抽取出来，并将其全部转换为小写
def textParse(bigString):
    #分隔bigString中的单词，分隔符为除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    #选出长度大于2的单词变为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]



