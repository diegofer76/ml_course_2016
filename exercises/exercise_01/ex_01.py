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

def chooseComponentsNumber(matrix, percent):
    print "\n---- Choosing components number ----"
    mat = np.matrix(matrix) * np.matrix(matrix).transpose() 
    U,S,V = np.linalg.svd(mat) 
    #print U.shape, S.shape, V.shape
    s_sum_all = sum(S)
    totalComponents = matrix.shape[1]
    num = totalComponents
    print "TotalComponents", totalComponents
    for i in range(totalComponents):
        if sum(S[0:i]) / s_sum_all >= percent :
            print "Nro components:",i ,"with variance =", sum(S[0:i]) / s_sum_all
            num = i
            break
    return num

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

def applyPCA(data, numComponents):
    print "\n---- Apply PCA ----"
    #pca = PCA(n_components=numComponents)
    pca = PCA(n_components=numComponents)
    #pca.fit(data)
    #cov_mat = pca.get_covariance()
    pcaData = pca.fit_transform(data)
    print pcaData.shape
    return pcaData  

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
    variance = 80
    numComponents = chooseComponentsNumber(data_train, float(variance) / 100)
    if numComponents == -1 : print "Invalid components number. Exit"; return
    
    pcaDataTrain = applyPCA(data_train, numComponents)
    pcaDataTest = applyPCA(data_test, numComponents)

    logreg = logisticRegression(data_train, classListTrain)
    #logreg.predict(data_test)
    print "Logistic Regression score: ", logreg.score(data_test, classListTest)

    logregPca = logisticRegression(pcaDataTrain, classListTrain)
    print "PCA (",variance,"%) Logistic Regression score: ", logregPca.score(pcaDataTest, classListTest)

    #logreg.predict_log_proba(data_test)
    #logreg.predict_proba(data_test)

if __name__ == "__main__":
    sys.exit(main())
