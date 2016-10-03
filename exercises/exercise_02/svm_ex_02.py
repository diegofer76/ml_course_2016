#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold

datFileName="data1.csv"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]

## Load CSV
def loadCsvData(fileName):
    raw_data = open(fileName, 'rb')
    rawData = pandas.read_csv(raw_data, delimiter=",", skiprows=1)
    return rawData.values

def getData(rawData):
    print "\n---- Getting data from File ----"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    print "lineNum:", lineNum
    print "colNum:", colNum

    data = np.array(rawData[0:lineNum, 0:colNum-1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    return [data, classList]

def getTrainAndTestData(data):
    print "\n---- Get Train and Test data ----"
    data_train = data[0:199]
    data_test = data[200:data.shape[0]]

    print "Data Train size:", data_train.shape 
    print "Data Test size:", data_test.shape
    return [data_train, data_test]

def getClassTrainTest(classList):
    print "\n---- Get Class Train and Test ----"
    classListTrain=classList[0:199]
    classListTest=classList[200:len(classList)]
    print len(classListTrain), len(classListTest)
    return [classListTrain, classListTest]

# A validacao externa deve ser 5-fold estratificado.
def externFolds(classList):
    [data_train, data_test] = getTrainAndTestData(data)
    [classListTrain, classListTest] = getClassTrainTest(classList)

    skf = cross_validation.StratifiedKFold(classList, n_folds=5)
    # modelo = fit(data_train, hmax)
	# ac = acuracia(modelo, data_test)


# In order to get hyperparameter
def internFolds(data, classList):
    [data_train, data_test] = getTrainAndTestData(data)
    [labelsTrain, labelsTest] = getLabelsTrainTest(classList)
    
    # MDOX =  fit (data_train, H)
    best_acc = 0
    for c in [2**-5, 2**-2, 1, 2**2, 2**5]:
        for gam in [2**-15, 2**-10, 2**-5, 1, 2**5]:
            svm = SVM.SVC(C = c, gamma = gam)
            svm.fit(data_train, labelsTrain)
            accuracy = svm.score(data_test, labelsTest)
        if accuracy > best_acc:
            best_acc = accuracy
    print("Best accuracy with 3-fold:", best_acc)
    # ACX = ACX + acuracia(mdox, data_test)
    # if acx > acxmax:
    #    acmax = acx
    #    hmax = h

def nanda():
    print 'nn'

def main(argv=None):
    if argv is None:
        arv = sys.argv
    rawdata = loadCsvData(dirPath + "/" + datFileName)
    [data, classList] = getData(rawdata)
    [data_train, data_test] = getTrainAndTestData(data)
    [classListTrain, classListTest] = getClassTrainTest(classList)

    skf = cross_validation.StratifiedKFold(classList, n_folds=5)
    for train_index, test_index in skf:
        new_data_train = data[train_index]
        print "len: " , len(new_data_train)

    print "-->", skf

if __name__ == "__main__":
    sys.exit(main())
