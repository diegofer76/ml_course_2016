#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
import math
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm as SVM
import  scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier

#from sklearn.neural_network import MLPClassifier

datFileName="secom.data"
labelsFileName="secom_labels.data"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]

def load_data(fileName):
    raw_data = open(fileName, 'rb')
    rawData = pandas.read_csv(raw_data, delimiter=" ")
    return rawData.values

def getData(rawData):
    #print "\n---- Getting data from File ----"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    data = np.array(rawData[0:lineNum, 0:colNum-1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    return [data, np.array(classList) ]

def getLabels(fileName):
    labelData = load_data(dirPath + "/" + fileName)
    labels = labelData[:,0].clip(min=0)
    return np.array(labels)


# In order to get hyperparameter
#def internFolds(data_train, data_test, labelsTrain, labelsTest):
#    acxmax = 0
#    c_max=0
#    gamma_max=0
#    for c in [2**-5, 2**-2, 1, 2**2, 2**5]:
#        for gamm in [2**-15, 2**-10, 2**-5, 1, 2**5]:
#            svm = SVM.SVC(C = c, gamma = gamm)
#            svm.fit(data_train, labelsTrain)
#            accuracy = svm.score(data_test, labelsTest)
#            if accuracy > acxmax:
#                acxmax = accuracy
#                c_max = c
#                gamma_max = gamm
#    return [acxmax, c_max, gamma_max]

def svm_intern_folds(data_train, data_test, labelsTrain, labelsTest):
    acxmax = 0
    c_max=0
    gamma_max=0
    for c in [2**(-5), 1, 2**(5), 2**(10)]:
        for gamm in [2**(-15), 2**(-10), 2**(-5), 1, 2**5]:
            svm = SVM.SVC(C = c, gamma = gamm)
            svm.fit(data_train, labelsTrain)
            accuracy = svm.score(data_test, labelsTest)
            if accuracy > acxmax:
                acxmax = accuracy
                c_max = c
                gamma_max = gamm
    return [acxmax, c_max, gamma_max]

def knn_intern_folds(data_train, data_test, labels_train, labels_test):
    acxmax = 0
    cores = 4
    k_value = 0
    for k in [1, 5, 11, 15, 21, 25]:
        knn = KNeighborsClassifier(n_neighbors = k, n_jobs = cores)
        knn.fit(data_train, labels_train)
        accuracy = knn.score(data_test, labels_test)
        #print "first acc:", accuracy
        if accuracy > acxmax:
            acxmax = accuracy
            k_value = k
            #print "accuracy: ", accuracy
    return [acxmax, k]

def neural_intern_folds(data_train, data_test, labelsTrain, labelsTest):
    # 10, 20, 30 e 40 neuronios na camada escondida.
    acxmax = 0
    cores = 4
    n_value = 0
    for n in [10, 20, 30, 40]:
        clf = MLPClassifier(hidden_layer_sizes=(n,))
        clf.fit(data_train, labels_train)
        accuracy = clf.score(data_test, labels_test)
        #print "first acc:", accuracy
        if accuracy > acxmax:
            acxmax = accuracy
            n_value = n
            print "accuracy: ", accuracy
    return [acxmax, n]

def rf_intern_folds(data_train, data_test, labelsTrain, labelsTest):
    # teste com mtry ou n_featrues = 10, 15, 20, 25 e ntrees = 100, 200, 300 e 400
    print ""

#def svm_rbf(data, labels):
#    final_accuracy = 0
#    skf = cross_validation.StratifiedKFold(labels, n_folds=5)
#    for train_index, test_index in skf:
#        new_data_train = data[train_index]
#        new_data_test = data[test_index]
#        new_labels_train = labels[train_index]
#        new_labels_test = labels[test_index]

#        acx = 0
#        skf_intern = cross_validation.StratifiedKFold(new_labels_train, n_folds=3)
#        for intern_train_index, intern_test_index in skf_intern:
#            intern_data_train = new_data_train[intern_train_index]
#            intern_data_test = new_data_train[intern_test_index]
#            intern_labels_train = new_labels_train[intern_train_index]
#            intern_labels_test = new_labels_train[intern_test_index]
#            [accuracy, c, gamma] = internFolds(intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
#
#            if accuracy > acx:
#                acx = accuracy
#                c_final = c
#                gamma_final = gamma
        ##print("acx", acx)
#        print("Valor Hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )
#        svm_model = SVM.SVC(C = c_final, gamma = gamma_final)
#        svm_model.fit(new_data_train, new_labels_train)
#        acc_5_fold = svm_model.score(new_data_test, new_labels_test)
#        acc_5_fold = svm_model.score(new_data_test, new_labels_test)
#        final_accuracy = final_accuracy + acc_5_fold
##        final_accuracy = final_accuracy + acx

#    final_accuracy = final_accuracy / 5
#    print("Acuracia media:%s" % final_accuracy)
#    print("Valor final hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )


## Data preprocessing
def data_preprocess(fileName):
    rawdata = load_data(dirPath + "/" + fileName)
    ## column mean
    column_mean = stats.nanmean(np.array(rawdata))
    ## Nan values index
    nan_indexes = np.where(np.isnan(rawdata))
    ## Replace Nan values
    rawdata[nan_indexes] = np.take(column_mean, nan_indexes[1])
    ## Standarize each column individually
    rawdata = (rawdata - np.mean(rawdata, axis=0)) / np.std(rawdata, axis=0)
    rawdata = np.nan_to_num(rawdata)
    return rawdata

def model_score(alg, data, labels):
    if 'svm'== alg:
        print "svm"
        [accuracy, c, gamma] = svm_intern_folds (intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
        if accuracy > acx:
            acx = accuracy
            c_final = c
            gamma_final = gamma

        print("Valor Hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )
        final_accuracy = final_accuracy + acx

    elif 'knn' == alg:
        print 'knn'


def run_svm_fold( data, labels):
    print "--- SVM ---"
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
            [accuracy, c, gamma] = svm_intern_folds (intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            if accuracy > acx:
                acx = accuracy 
                c_final = c
                gamma_final = gamma
        #print("Valor Hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )
        final_accuracy = final_accuracy + acx

    final_accuracy = final_accuracy / 5
    print("Acuracia media:%s" % final_accuracy)
    print("Valor final hiperparametros (C=%s, Gamma=%s)" % (c_final, gamma_final) )

def run_knn_folds( data, labels):
    print "--- Knn ---"
    final_accuracy = 0
    k_final = 0
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
            [accuracy, k] = knn_intern_folds (intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            if accuracy > acx:
                acx = accuracy
                k_final = k
        #print("Valor K neighbors (K=%s)" % (k_final) )
        final_accuracy = final_accuracy + acx

    final_accuracy = final_accuracy / 5
    print("Acuracia media:%s" % final_accuracy)
    print("Valor final K (K=%s)" % (k_final))

def run_neural_folds( data, labels):
    print "--- Neural Network ---"
    final_accuracy = 0
    k_final = 0
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
            [accuracy, n] = neural_intern_folds (intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            if accuracy > acx:
                acx = accuracy
                n_final = n
        print("Valor n neurons (N=%s)" % (n_final) )
        final_accuracy = final_accuracy + acx

    final_accuracy = final_accuracy / 5
    print("Acuracia media:%s" % final_accuracy)
    print("Valor final neurons (N=%s)" % (n_final))

def main(argv=None):
    if argv is None:
        arv = sys.argv

    ## Data pre-processing    
    data = data_preprocess(datFileName)
    labels = getLabels(labelsFileName)
    labels = np.array(list(labels[:data.shape[0]]))

    ## kNN , PCA com 80% da variancia
    #model_score('knn', data, labels)
    run_knn_folds(data, labels)

    ## SVM RBF 
    run_svm_fold(data, labels)

    ## Neural network
#    run_neural_folds(data, labels)

if __name__ == "__main__":
    sys.exit(main())
