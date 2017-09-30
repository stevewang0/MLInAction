# encoding: utf-8
'''
Created on Sep 26, 2017
使用文档数据测试朴素贝叶斯模块

@author: steve.wang
'''

import sys
sys.path.append(r'../../../module') #将algo模块所在路径添加进来
import Bayes
from numpy import *

def loadDataSet():
    #定义6个文档样例
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #样本类型：1代表侮辱性文字， 0代表正常言论
    return postingList,classVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = Bayes.createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        #trainMat.append(Bayes.setOfWords2Vec(myVocabList, postinDoc))
        trainMat.append(Bayes.bagOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = Bayes.trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    #thisDoc = array(Bayes.setOfWords2Vec(myVocabList, testEntry))
    thisDoc = array(Bayes.bagOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', Bayes.classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    #thisDoc = array(Bayes.setOfWords2Vec(myVocabList, testEntry))
    thisDoc = array(Bayes.bagOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', Bayes.classifyNB(thisDoc, p0V, p1V, pAb)

testingNB()
