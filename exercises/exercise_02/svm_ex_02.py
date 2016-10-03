#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm as SVM

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
    return [data, np.array(classList) ]

def getTrainAndTestData(data):
    print "\n---- Get Train and Test data ----"
    data_train = data[0:199]
    data_test = data[200:data.shape[0]]

    print "Data Train size:", data_train.shape 
    print "Data Test size:", data_test.shape
    return [data_train, data_test]

def getLabelsTrainTest(classList):
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
def internFolds(data_train, data_test, labelsTrain, labelsTest):
    
    # MDOX =  fit (data_train, H)
    acxmax = 0
    c_max=0
    gamma_max=0
    for c in [2**-5, 2**-2, 1, 2**2, 2**5]:
        for gamm in [2**-15, 2**-10, 2**-5, 1, 2**5]:
            svm = SVM.SVC(C = c, gamma = gamm)
            svm.fit(data_train, labelsTrain)
            accuracy = svm.score(data_test, labelsTest)
            if accuracy > acxmax:
                acxmax = accuracy
                c_max = c
                gamma_max = gamm
	return [acxmax, c_max, gamma_max]
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
    [data, labels] = getData(rawdata)
    [data_train, data_test] = getTrainAndTestData(data)
    [labels_train, labels_test] = getLabelsTrainTest(labels)
    final_accuracy = 0

    skf = cross_validation.StratifiedKFold(labels, n_folds=5)
    for train_index, test_index in skf:
        new_data_train = data[train_index]
        new_data_test = data[test_index]
        new_labels_train = labels[train_index]
        new_labels_test = labels[test_index]
        #print "len: " , len(new_data_train)
        
        skf_intern = cross_validation.StratifiedKFold(new_labels_train, n_folds=3)
        mean_acc = 0
        for intern_train_index, intern_test_index in skf_intern:
            intern_data_train = new_data_train[intern_train_index]	
            intern_data_test = new_data_train[intern_test_index]	
            intern_labels_train = new_labels_train[intern_train_index]	
            intern_labels_test = new_labels_train[intern_test_index]
            print "len inter:", len(intern_data_train)	
            [accuracy, c_final, gamma_final ] = internFolds(intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            mean_acc = mean_acc + accuracy
            print("C and Gamma values with 3-fold:", c_final, gamma_final)

#        svm = SVM.SVC(C = c_final, gamma = gamma_final)
#        svm.fit(data_train, labels_train)
#        mean_acc = svm.score(data_test, labels_test)
#        print("Final Accuracy:", final_accuracy)
#        mean_acc = mean_acc / 3
#        print("Mean accuracy with 3-fold:", mean_acc)
        #print("C and Gamma values with 3-fold:", c_final, gamma_final)

    svm = SVM.SVC(C = c_final, gamma = gamma_final)
    svm.fit(data, labels)
    final_accuracy = svm.score(data, labels)
    print("Final Accuracy:", final_accuracy)

if __name__ == "__main__":
    sys.exit(main())
