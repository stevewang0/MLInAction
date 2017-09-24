# encoding: utf-8
'''
Created on Sep 18, 2017
使用matplotlib绘制决策树

@author: steve.wang
'''

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8") #定义非叶子结点的样式
leafNode = dict(boxstyle = "round4", fc = "0.8") #定义叶子结点的样式
arrow_args = dict(arrowstyle = "<-") #定义节点之间连接线的样式

#绘制一个结点，nodeTxt为结点显示文本，centerPt为文本起始位置，parentPt为箭头的起始位置，nodeType为结点框的样式
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #使用annotate()方法可以很方便地添加文字注释
    # 第一个参数是注释的内容  
    # xy设置箭头尖的坐标  
    # xytext设置注释内容显示的起始位置  
    # arrowprops 用来设置箭头样式
    # bbox用来设置节点框的样式
    # xycoords and textcoords 是坐标xy与xytext的说明，若textcoords=None，则默认textNone与xycoords相同，若都未设置，默认为data
    # va/ha设置节点框中文字的位置，va取值为(u'top', u'bottom', u'center', u'baseline')，ha取值为(u'center', u'right', u'left')
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',
                            xytext = centerPt, textcoords = 'axes fraction',
                            va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

def createPlot_old():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

#获取当前树中的叶子结点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0] #获取当前树myTree中第一个key，即该树的根节点
    secondDict = myTree[firstStr] #获取第一个key对应的内容，即根节点下的子树
    for key in secondDict.keys(): #根节点对应的各个分支，依次循环
        #type()就是一个最实用又简单的查看数据类型的方法。
        #type()是一个内建的函数，调用它就能够得到一个反回值，从而知道想要查询的对像类型信息。
        if type(secondDict[key]).__name__ == 'dict': #如果该子树还是一棵树，递归调用函数getNumLeafs()，获取子树的叶子结点数
            numLeafs += getNumLeafs(secondDict[key])
        else: #如果是叶子结点，则叶子数加1
            numLeafs += 1
    return numLeafs #返回当前树中叶子结点的个数

#获取当前树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0] #获取树的根节点
    secondDict = myTree[firstStr] #获取树的子树
    for key in secondDict.keys(): #根节点对应的各个分支，依次循环
        #如果该子树还是一棵树，递归调用函数getTreeDepth()，获取子树的深度
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: #如果是叶子结点，则返回1
            thisDepth = 1
        if thisDepth > maxDepth: #更新最大深度变量值
            maxDepth = thisDepth
    return maxDepth #返回最大深度

#使用字典类型定义了两棵树，返回指定的树
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':\
                                                 {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers':\
                                                 {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#显示文本，在坐标点cntrPt和parentPt连接线上的中点，显示文本txtString
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0] #计算x坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1] #计算y坐标
    createPlot.ax1.text(xMid, yMid, txtString) #在(xMid, yMid)处显示txtString


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree) #获取样本树的叶子结点数目
    depth = getTreeDepth(myTree) #获取样本树的深度
    firstStr = myTree.keys()[0] #获取样本树的根结点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, \
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) #绘制文本信息nodeTxt
    plotNode(firstStr, cntrPt, parentPt, decisionNode) #绘制根结点
    secondDict = myTree[firstStr] #获取各个子树
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys(): #循环遍历各个子树
        if type(secondDict[key]).__name__ == 'dict': #如果包含的是子树，递归调用plotTree绘制结点
                plotTree(secondDict[key], cntrPt, str(key))
        else:
                plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) #绘制叶子结点
                plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) #绘制文字注释
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD #调整Y轴的坐标值

#创建figure并绘制树inTree
def createPlot(inTree):
    #Matplotlib 里的常用类的包含关系为 Figure -> Axes -> (Line2D, Text, etc.)
    #一个Figure对象可以包含多个子图(Axes)，在matplotlib中用Axes对象表示一个绘图区域，可以理解为子图。
    fig = plt.figure(1, facecolor = 'white') #定义一个figure对象，背景色设置为全白
    fig.clf() #清楚figure中的内容
    axprops = dict(xticks = [], yticks = []) 
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops) #在图表fig中创建一个子图ax1
    plotTree.totalW = float(getNumLeafs(inTree)) #获取样本树的叶子结点数目作为plotTree的宽度
    plotTree.totalD = float(getTreeDepth(inTree)) #获取样本树的深度作为plotTree的深度
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '') #依次绘制整棵决策树
    plt.show()
