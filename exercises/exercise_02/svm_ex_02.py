#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
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

def main(argv=None):
    if argv is None:
        arv = sys.argv
    rawdata = loadCsvData(dirPath + "/" + datFileName)
    [data, classList] = getData(rawdata)
    [data_train, data_test] = getTrainAndTestData(data)
    [classListTrain, classListTest] = getClassTrainTest(classList)

if __name__ == "__main__":
    sys.exit(main())
