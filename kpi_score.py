# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import sklearn
import os
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import xlrd
import xlutils
import xlwt

#自动把空格行清除
#返回name,dataset(2维数组),label(1维数组)
def loadexceldata(filename):
    fr = xlrd.open_workbook(filename)
    fout = open("kpi_data.txt", "w")
    list = fr.sheets()
    for line in list:
        nums = line.nrows
        name = []
        trainingLabels = []
        trainingSet = []
        for i in range(1,line.nrows):
            a = line.row(i)[1].value
            try:
                temp = float(a)
                if temp == 0:
                    raise ValueError
            except ValueError:
                continue
            name.append(line.row(i)[0].value)
            temp = [line.row(i)[j].value for j in range(1,6)]
            trainingSet.append(temp)
    base1 = [2.80, 50.00, 30.00, 0.1, 0.40]
    trainingSet.append(base1)
    min_max_scler = preprocessing.MinMaxScaler()
    nums = len(trainingSet)-1
    X = min_max_scler.fit_transform(trainingSet)
    base = X[nums][0] + X[nums][1] + X[nums][2] + X[nums][3]
    for i in range(nums):
        score = X[i][0] + X[i][1] + X[i][2] + X[i][3]
        if score >= base:
            trainingLabels.append(1)
        else:
            trainingLabels.append(0)
    for i in range(0,nums):
        for j in range(4):
            X[i][j] = float(X[i][j])
            fout.write(str(X[i][j]) + "\t")
            j += 1
        fout.write(str(trainingLabels[i]) + "\n")
        i += 1
    fout.close()
    return name,trainingLabels[:124],trainingSet[:124]

#划分数据集
def splitDataSet(fileName, split_size,outdir):
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    fr = open(fileName,'r')#open fileName to read
    #num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line) #get a seq and set len=numLine
    np.random.shuffle(arr) #generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line+1) / split_size #size of each split sets
    split_all = []; each_split = []
    count_num = 0; count_split = 0  #count_num 统计每次遍历的当前个数
                                    #count_split 统计切分次数
    for i in range(len(list_all)): #遍历整个数字序列
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  #输出每一份数据
            split_all.append(each_split) #将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all

def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []; labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split('\t') #remove '\n'
        for i in range(0, len(curLine)-1):
            lineArr.append(float(curLine[i])) #get all feature from inpurfile
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1])) #last one is class lable
    fr.close()
    return dataMat,labelMat


#对数据集进行采样
def underSample(datafile): #只针对一个数据集的下采样
    dataMat,labelMat = loadDataSet(datafile) #加载数据
    pos_num = 0; pos_indexs = []; neg_indexs = []
    for i in range(len(labelMat)):#统计正负样本的下标
        if labelMat[i] == 1:
            pos_num +=1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]
        outfile.append(pos_line)
        if i < (len(labelMat)-pos_num):
            neg_line= onefile[neg_indexs[i]]
            outfile.append(neg_line)
    return outfile #输出单个数据集采样结果

#测试集和训练集
def generateDataset(datadir,outdir): #从切分的数据集中，对其中九份抽样汇成一个,\
    #剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = []; test_all = [];cross_now = 0
    for eachfile1 in listfile:
        train_sets = []; test_sets = [];
        cross_now += 1 #记录当前的交叉次数
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:#对其余九份欠抽样构成训练集
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        #将训练集和测试集文件单独保存起来
        with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:
                    test_sets.append(each_testline)
            for oneline_test in test_sets:
                fw_test.write(oneline_test) #输出测试集
            test_all.append(test_sets)#保存训练集
        with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
            for oneline_train in train_sets:
                oneline_train = oneline_train
                fw_train.write(oneline_train)#输出训练集
            train_all.append(train_sets)#保存训练集
    return train_all, test_all

#逻辑回归
def loadData(stra):
    str_train = "sample_data1/train_"+ str(stra) + ".datasets"
    str_test = "sample_data1/test_"+ str(stra) + ".datasets"
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    fr = open(str_train)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        train_data.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        train_label.append(int(lineArr[4]))
    fr.close()

    fr = open(str_test)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        test_data.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        test_label.append(int(lineArr[4]))
    fr.close()
    return train_data,train_label,test_data,test_label

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix

    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult

    return weights

def Cost_Function_one(X,Y,theta):
    sumOfErrors = 0
    for i in range(len(X)):
        itemp = i % len(X)
        xi = X[itemp]
        hi = Hypothesis(theta,xi)
        if Y[itemp] == 1:
            error = Y[itemp] * math.log(hi)
        elif Y[itemp] == 0:
            error = (1-Y[itemp]) * math.log(1-hi)
            sumOfErrors += error
    cons = - 1/len(X)
    Ji = cons * sumOfErrors
    return Ji

def Hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return sigmoid(z)

def GetResult():
    accuracy = [0,0,0,0,0,0]
    accuracy[0] = 0
    weight = []
    for i in range(1,6):
        #accuracy[0] = 0
        train_data, train_label, test_data, test_label = loadData(i)
        weights = gradAscent(train_data, train_label)
        a = testLogRegres(weights,test_data,test_label)
        accuracy[i] = a
        if accuracy[i] > accuracy[i-1]:
            weight = weights
        #print("损失函数%s"%Cost_Function(train_data,train_label,weights,1000))
        #print("正确率%s"%testLogRegres(weights,test_data,test_label))
        #print(weights)
    return weight


    #plotBestFit(weights)

def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        x = test_x[i][0]*weights[0][0] +  test_x[i][1]*weights[1][0] +  test_x[i][2]*weights[2][0] + test_x[i][3]*weights[3][0] + test_x[i][4]*weights[4][0]
        predict = sigmoid(x)> 0.5
        if predict == True:
            predict1 = 1
        else:
            predict1 = 0
        if predict1 == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

def testLogRegres1(weights, test_x, test_y, name):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    fout = open("output1","w")
    for i in range(numSamples):
        x = test_x[i][0]*weights[0][0] +  test_x[i][1]*weights[1][0] +  test_x[i][2]*weights[2][0] + test_x[i][3]*weights[3][0] + test_x[i][4]*weights[4][0]
        predict = float(sigmoid(x))
        score = predict * 100
        string = str(name[i]) + "\t" + str(score) + "\n"
        fout.write(string)
        print("%s,           %s"%(name[i],predict))
        predict = sigmoid(x) > 0.5
        if predict == True:
            predict1 = 1
        else:
            predict1 = 0
        if predict1 == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    fout.write("本次测试的正确率为：%s" %accuracy )
    return accuracy

##输入name和评分结果
def write_excel(weights,test_x,test_y,name):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    fprint = xlwt.Workbook()
    '''
    创建第一个sheet:
      sheet1
    '''
    sheet1 = fprint.add_sheet(u'结果')  # 创建sheet
    row0 = [u"序号",u'渠道名称', u'评分']
    # 生成第一行
    first_col = sheet1.col(1)
    first_col.width = 256*30


    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])

    for i in range(numSamples):
        x = weights[0][0] + test_x[i][0] * weights[1][0] + \
            test_x[i][1] * weights[2][0] + test_x[i][2] * weights[3][0] + test_x[i][3] * weights[4][0]
        predict = float(sigmoid(x))
        score = predict * 100
        ##写入excel
        sheet1.write(i+1,0,i)
        sheet1.write(i+1,1,name[i])
        sheet1.write(i+1,2,score)
        print("%s,           %s" % (name[i], predict))
        predict = sigmoid(x) > 0.5
        if predict == True:
            predict1 = 1
        else:
            predict1 = 0
        if predict1 == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    if os.path.exists("result.xlsx"):
        os.remove("result.xlsx")
    fprint.save("result.xlsx")
    return accuracy




name,labelSet,dataSet =loadexceldata("2017_05.xlsx")
splitall = splitDataSet("kpi_data.txt",6,"split")
train_all,test_all = generateDataset("split","sample_data1")
weight = GetResult()
test_x,test_y = loadDataSet("kpi_data.txt")
print(write_excel(weight,test_x,test_y,name))












