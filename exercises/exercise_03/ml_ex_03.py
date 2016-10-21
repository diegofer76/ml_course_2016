#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
import math
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm as SVM
import  scipy.stats as stats

datFileName="secom.data"
labelsFileName="secom_labels.data"
#datFileName="../exercise_02/data1.csv"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]

def load_data(fileName):
    raw_data = open(fileName, 'rb')
    rawData = pandas.read_csv(raw_data, delimiter=" ")
    print "load_data", rawData.shape
    return rawData.values

def getData(rawData):
    #print "\n---- Getting data from File ----"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    data = np.array(rawData[0:lineNum, 0:colNum-1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    return [data, np.array(classList) ]

## Data preprocessing
def data_preprocess(fileName):
    rawdata = load_data(dirPath + "/" + fileName)
    ## column mean
    column_mean = stats.nanmean(rawdata, axis=0)
    ## Nan values index
    nan_indexes = np.where(np.isnan(rawdata))
    ## Replace Nan values
    rawdata[nan_indexes] = np.take(column_mean, nan_indexes[1])
    ## Standarize each column individually
    rawdata = (rawdata - np.mean(rawdata, axis=0)) / np.std(rawdata, axis=0)
    rawdata = np.nan_to_num(rawdata)

    np.savetxt("foo.csv", rawdata, delimiter=",")
    csv_data = load_data("foo.csv")
    return csv_data
    #return np.array(csv_data)

def getLabels(fileName):
    labelData = load_data(dirPath + "/" + fileName)
    labels = labelData[:,0].clip(min=0)
    return np.array(labels)


# In order to get hyperparameter
def internFolds(data_train, data_test, labelsTrain, labelsTest):
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

def svm_rbf(data, labels):
    final_accuracy = 0
    skf = cross_validation.StratifiedKFold(labels, n_folds=5)
    for train_index, test_index in skf:
        new_data_train = data[train_index]
        new_data_test = data[test_index]
        new_labels_train = labels[train_index]
        new_labels_test = labels[test_index]

        acx = 0
        skf_intern = cross_validation.StratifiedKFold(new_labels_train, n_folds=3)
        for intern_train_index, intern_test_index in skf_intern:
            intern_data_train = new_data_train[intern_train_index]	
            intern_data_test = new_data_train[intern_test_index]	
            intern_labels_train = new_labels_train[intern_train_index]	
            intern_labels_test = new_labels_train[intern_test_index]
            [accuracy, c, gamma] = internFolds(intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            if accuracy > acx:
                acx = accuracy 
                c_final = c
                gamma_final = gamma
        #print("acx", acx)        
        print("Valor Hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )
        svm_model = SVM.SVC(C = c_final, gamma = gamma_final)
        svm_model.fit(new_data_train, new_labels_train)
        acc_5_fold = svm_model.score(new_data_test, new_labels_test)
        acc_5_fold = svm_model.score(new_data_test, new_labels_test)
        final_accuracy = final_accuracy + acc_5_fold
        #final_accuracy = final_accuracy + acx

    final_accuracy = final_accuracy / 5
    print("Acuracia media:%s" % final_accuracy)
    print("Valor final hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )


def main(argv=None):
    if argv is None:
        arv = sys.argv

    ## Data pre-processing    
    data = data_preprocess(datFileName)
    labels = getLabels(labelsFileName)
    labels = labels[:data.shape[0]]
    
    print "---------------" 
    print "data shape:", data.shape
    print "label shape:", labels.shape

    ## kNN , PCA com 80% da variancia

    ## SVM RBF 
    #svm_rbf(data, labels)


if __name__ == "__main__":
    sys.exit(main())
