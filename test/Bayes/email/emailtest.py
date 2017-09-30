# encoding: utf-8
'''
Created on Sep 26, 2017
使用邮件数据测试朴素贝叶斯模块

@author: steve.wang
'''

import sys
sys.path.append(r'../../../module') #将algo模块所在路径添加进来
import Bayes
from numpy import *
#正则表达式本身是一种小型的、高度专业化的编程语言，而在python中，通过内嵌集成re模块，
#可以直接调用来实现正则匹配。正则表达式模式被编译成一系列的字节码，然后由用C编写的匹配引擎执行。
import re

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26): #将文件夹spam和ham下所有文本文件解析出来
        #从对应文本文件中读出字符串，将其解析为单词列表
        wordList = Bayes.textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList) #将当前文本的词汇列表添加到docList变量中
        fullText.extend(wordList) #将当前文本的所有单词追加到fullText变量中
        classList.append(1) #分类列表变量classList中增加一个1类信息
        wordList = Bayes.textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #分类列表变量classList中增加一个0类信息
    vocabList = Bayes.createVocabList(docList) #获取docList中所有出现过的单词的词汇表
    trainingSet = range(50) #创建拥有50个元素的list变量，存储0-49个数字，对应spam与ham目录下所有文本
    testSet = []
    for i in range(10): #从0-9循环，产生10个测试样本id
        #uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内。
        #在[0, 50)之间产生一个随机整数
        randIndex = int(random.uniform(0, len(trainingSet)))
        print randIndex
        #将trainingSet中对应训练样本id添加到测试集testSet中，并从trainingSet中删除该id
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    #在40个训练样本中，逐个文本循环处理，获取1类和0类各个单词出现的概率及1类样本的概率
    for docIndex in trainingSet:
        #获取当前文档中单词在词汇表vocabList是否出现的列表，添加到列表变量trainMat中
        trainMat.append(Bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        #将对应文档的分类信息添加到trainClasses中
        trainClasses.append(classList[docIndex])
    #获取训练样本中1类和0类各个词汇的出现概率，及所有样本中1类样本所占概率
    p0V, p1V, pSpam = Bayes.trainNB0(array(trainMat), array(trainClasses))
    print classList
    errorCount = 0
    #使用10个测试样本，对贝叶斯分类效果进行检测
    for docIndex in testSet:
        #获取当前测试样本中单词在词汇表vocabList是否出现的列表
        wordVector = Bayes.setOfWords2Vec(vocabList, docList[docIndex])
        #使用贝叶斯分类器对当前测试样本进行分类，判断分类结果是否正确
        if Bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet) #打印出分类错误率
    
spamTest()
