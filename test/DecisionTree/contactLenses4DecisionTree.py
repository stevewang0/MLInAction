# encoding: utf-8
'''
Created on Sep 20, 2017
使用隐形眼镜样本集对决策树进行测试

@author: steve.wang
'''
import sys
sys.path.append(r'../../module') #将algo模块所在路径添加进来
import DecisionTreePlotter
import DecisionTree
import pickle

#持久性就是指保持对象，甚至在多次执行同一程序之间也保持对象。
#pickle是为了序列化/反序列化一个对象的，可以把一个对象持久化存储。
#pickle 模块提供了以下函数对： dumps(object) 返回一个字符串，它包含一个 pickle 格式的对象；
#loads(string) 返回包含在 pickle 字符串中的对象；
#dump(object, file) 将对象写到文件，这个文件可以是实际的物理文件，但也可以是任何类似于文件的对象，
#这个对象具有 write() 方法，可以接受单个的字符串参数； load(file) 返回包含在 pickle 文件中的对象。
          
def storeTree(inputTree, filename):        
        fw = open(filename, 'w')
        pickle.dump(inputTree, fw) #将输入的决策树写入文件保存起来
        fw.close()

def grabTree(filename):
        fr = open(filename)
        return pickle.load(fr) #返回文件中保存的对象

fr = open('lenses.txt') #打开样本集文件
#strip() 方法用于移除字符串头尾指定的字符（默认为空格）
#split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串
#str.split(str="", num=string.count(str))
#readlines()从文件中一行一行地读数据，返回一个列表；读取的行数据包含换行符
#从样本集文件中读取所有的行，用换行符分开，去除每行行首和行末的空格，保存到列表变量lenses中
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] #定义样本集的特征集
lensesTree = DecisionTree.createTree(lenses, lensesLabels) #调用模块DecisionTree的函数createTree对样本集产生决策树
print lensesTree
storeTree(lensesTree, 'DecisionTreeStorage.txt') #将决策树保存到文件中

inTree = grabTree('DecisionTreeStorage.txt') #从文件中加载决策树
DecisionTreePlotter.createPlot(inTree) #调用模块DecisionTreePlotter的函数createPlot绘制产生的决策树

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] #定义样本集的特征集
print DecisionTree.classify0(inTree, lensesLabels, ['young','hyper','no','normal'])
print DecisionTree.classify(inTree, lensesLabels, ['young','hyper','no','reduced'])
