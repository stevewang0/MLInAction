# encoding: utf-8
'''
Created on Sep 26, 2017
使用RSS数据测试朴素贝叶斯模块

@author: steve.wang
'''

import sys
sys.path.append(r'../../../module') #将algo模块所在路径添加进来
import Bayes
from numpy import *
import operator
import feedparser

#vocabList：词汇表
#fullText：测试文本
#从测试文本fullText查找出在词汇表vocabList中存在的单词出现的次数，返回出现次数最多的前30个单词信息
def calcMostFreq(vocabList, fullText):    
    freqDict = {} #定义字典变量，存储各个单词出现的次数
    for token in vocabList: #按词汇表中每个单词依次循环，找出文本fullText中各个单词出现的次数
        freqDict[token] = fullText.count(token) #找出token代表的单词在fullText中出现的次数，存储到freqDict中
    #对freqDict中元素按照第二个变量进行降序排序    
    sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30] #返回前30个元素

#feed1/0为两个RSS源
#获取两个RSS源中词汇表及各自词汇出现概率信息
def localWords(feed1, feed0):
    docList = [] #以二维数组形式存储所有样本的词汇表
    classList = [] #存储所有样本的类别信息
    fullText = [] #以一维数组形式存储所有样本的词汇表
    minLen = min(len(feed1['entries']), len(feed0['entries'])) #获取两个RSS源的最小长度
    for i in range(minLen):
        #解析feed1['entries'][i]['summary']，将长度大于2的单词提取出来，并全转换为小写
        wordList = Bayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList) #将该样本词汇添加到docList中
        fullText.extend(wordList) #将该样本词汇追加到fullText中
        classList.append(1) #将样本类别信息添加到classList
        wordList = Bayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = Bayes.createVocabList(docList) #获取docList中所有不重复的单词列表
    #由于语言中大部分都是冗余和结构辅助性内容，导致词汇表中一小部分单词却占据了所有文本用词的一大部分。需要去除冗余词汇。
    #另一个常用的方法是不仅移除高频词，同时从某个预定词表中移除结构上的辅助词。该词表称为停用词表（stop word list）。
    top30Words = calcMostFreq(vocabList, fullText) #获取在fullText中出现次数最多的30个词汇信息
    for pairW in top30Words: #从词汇表vocabList中去除出现次数最多的30个单词
       if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen); #定义列表变量存储训练样本id
    print 'minLen : %d' % minLen
    if minLen < 20:
        print 'the len is too small.'
    testSet = [] #用于存储测试样本id
    for i in range(20): #从训练样本中随机获取20个样本信息作为测试样本集，并从训练样本中去除这些样本
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    #从文本样本集中获取训练样本集，将相关文本样本的词汇出现次数信息存储到矩阵trainMat中，样本分类信息存储到trainClasses中
    for docIndex in trainingSet:
        #获取样本docList[docIndex]在词汇表vocabList中各个单词出现次数情况
        trainMat.append(Bayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        #获取当前样本的分类信息classList[docIndex]
        trainClasses.append(classList[docIndex])
    #通过贝叶斯分类器对训练样本进行学习
    #获取两个类别各自单词的出现频率，以及样本集的概率
    p0V, p1V, pSpam = Bayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    #使用测试样本集对学习结果进行测试
    for docIndex in testSet:
        #获取样本docList[docIndex]在词汇表vocabList中各个单词出现次数情况
        wordVector = Bayes.bagOfWords2Vec(vocabList, docList[docIndex])
        #对当前测试样本进行分类，判断是否与已知类型相同
        if Bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet) #打印出错误率
    return vocabList, p0V, p1V #返回词汇表和各个词汇的出现概率

#获取ny, sf中单词出现概率大于某个阈值的单词，按降序排序将其打印出来
def getTopWords(ny, sf):
    #获取两个RSS源ny, sf中词汇表及各自词汇出现概率信息
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    #对两个类别对应的单词概率进行判断，如果概率大于-5.0，将对应词汇和概率都存储起来
    for i in range(len(p0V)):
        if p0V[i] > -5.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -5.0:
            topNY.append((vocabList[i], p1V[i]))
    #lambda作为一个表达式，定义了一个匿名函数
    #g = lambda x:x+1  x为入口参数，x+1为函数体
    #topSF为二维列表，pair[1]指向词汇对应的概率，即按照概率值进行降序排序
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse = True)
    print "SF**SF**SF**"
    for item in sortedSF:#打印出各项单词
        print item[0]
    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse = True)
    print "NY**NY**NY**"
    for item in sortedNY:
        print item[0]
    
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = localWords(ny, sf)
vocabList, pSF, pNY = localWords(ny, sf)
getTopWords(ny, sf)
