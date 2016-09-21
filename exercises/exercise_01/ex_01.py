#!/usr/bin/python

import sys,os,csv
import pandas
import numpy
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# pca.fit(X)
# PCA(copy=True, n_components=2, whiten=False)

datFileName="data1.csv"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]

## Load CSV
def loadCsvData(fileName):
    raw_data = open(fileName, 'rb')
    rawData = pandas.read_csv(raw_data, delimiter=",", skiprows=1)
    #print(rawData.shape)
    return rawData.values

def getData(rawData):
    print "\n---- Getting data ----"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    print "lineNum:", lineNum
    print "colNum:", colNum

    data = np.array(rawData[0:lineNum, 0:colNum-1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    return [data, classList]

 

def chooseComponentsNumber(matrix):
    matrix = np.matrix(matrix) * np.matrix(matrix).transpose() 
    U,S,V = np.linalg.svd(matrix) 
    #print U.shape, S.shape, V.shape
    s_sum_all = sum(S)
    for i in range(len(S)):
        if sum(S[0:i]) / s_sum_all >= 0.8 :
            print "Nro components:",i ,"with variance =", sum(S[0:i]) / s_sum_all
            break

def meanNormalization(rawdata):
    for i in range(rawdata.shape[0]):
        np.mean(data[i])
    return data

def getTrainAndTestData(data):
    print "\n---- Get Train and Test data ----"
    data_train = data[0:200]
    data_test = data[200:data.shape[0]]

    print "Data Train size:", data_train.shape 
    print "Data Test size:", data_test.shape
    return [data_train, data_test]

def getClassTrainTest(classList):
    print "\n---- Get Class Train and Test ----"
    classListTrain=classList[0:200]
    classListTest=classList[200:len(classList)]
    print len(classListTrain), len(classListTest)
    return [classListTrain, classListTest]

def applyPCA(data):
    print "\n---- Apply PCA ----"
    pca = PCA(n_components=80)
    pca.fit(data)
    #print "params" ,  pca.get_params()
    cov_mat = pca.get_covariance()
    #print cov_mat.shape
    #print(pca.explained_variance_ratio_)
    print data.shape
    return data  

def logisticRegression(data, classList):
    print "\n ---- Logistic Regression ----"
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(data, classList) 
    return logreg

def main(argv=None):
    if argv is None:
        arv = sys.argv
    rawdata = loadCsvData(dirPath + "/" + datFileName)
    [data, classList] = getData(rawdata)
    [data_train, data_test] = getTrainAndTestData(data)
    [classListTrain, classListTest] = getClassTrainTest(classList)

    #pcaData = applyPCA(data)
    pcaData = applyPCA(data_train)
    chooseComponentsNumber(data_train)
    logreg = logisticRegression(data_train, classListTrain)
    #logreg.predict(data_test)
    print "Logistic Regression score: ", logreg.score(data_test, classListTest)

    logregPca = logisticRegression(pcaData, classListTrain)
    print "PCA Logistic Regression score: ", logreg.score(data_test, classListTest)

    #logreg.predict_log_proba(data_test)
    #logreg.predict_proba(data_test)

if __name__ == "__main__":
    sys.exit(main())
