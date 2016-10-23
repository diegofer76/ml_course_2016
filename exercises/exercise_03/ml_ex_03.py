#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn import svm as SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

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
        if accuracy > acxmax:
            acxmax = accuracy
            k_value = k
    return [acxmax, k]

def neural_intern_folds(data_train, data_test, labels_train, labels_test):
    # 10, 20, 30 e 40 neuronios na camada escondida.
    acxmax = 0
    cores = 4
    n_value = 0
    for n in [10, 20, 30, 40]:
        clf = MLPClassifier(hidden_layer_sizes=(n,), solver='lbfgs')
        clf.fit(data_train, labels_train)
        accuracy = clf.score(data_test, labels_test)
        if accuracy > acxmax:
            acxmax = accuracy
            n_value = n
    return [acxmax, n]

def rf_intern_folds(data_train, data_test, labels_train, labels_test):
    # teste com mtry ou n_featrues = 10, 15, 20, 25 e ntrees = 100, 200, 300 e 400
    acxmax = 0
    n_feats = 0
    n_trees = 0
    for feat in [10, 15, 20, 25]:
        for trees in [100, 200, 300, 400]:
            clf = RandomForestClassifier (max_features = feat, n_estimators = trees)
            clf.fit(data_train, labels_train)
            accuracy = clf.score(data_test, labels_test)
            #print "first acc:", accuracy
            if accuracy > acxmax:
                acxmax = accuracy
                n_feats = feat
                n_trees = trees
    return [acxmax, n_feats, n_trees]

def gbm_intern_folds(data_train, data_test, labels_train, labels_test):
    ##  numero de arvores = 30, 70, e 100, com learning rate de 0.1 e 0.05, e profundidade da arvore=5.
    acxmax = 0
    n_learn_rate = 0
    n_trees = 0
    depth_tree = 5
    for trees in [30, 70, 100]:
        for learn_rate in [0.1, 0.05]:
            clf = GradientBoostingClassifier (n_estimators = trees, learning_rate = learn_rate, max_depth = depth_tree)
            clf.fit(data_train, labels_train)
            accuracy = clf.score(data_test, labels_test)
            #print "first acc:", accuracy
            if accuracy > acxmax:
                acxmax = accuracy
                n_trees = trees
                n_learn_rate = learn_rate
    return [acxmax, n_learn_rate, n_trees]

## Data preprocessing
def data_preprocess(fileName):
    rawdata = load_data(dirPath + "/" + fileName)
    ## column mean
    column_mean = np.nanmean(np.array(rawdata), axis=0)
    ## Nan values index
    nan_indexes = np.where(np.isnan(rawdata))
    ## Replace Nan values
    rawdata[nan_indexes] = np.take(column_mean, nan_indexes[1])
    ## Standarize each column individually
    rawdata = (rawdata - np.mean(rawdata, axis=0)) / np.std(rawdata, axis=0)
    rawdata = np.nan_to_num(rawdata)
    return rawdata

def run_folds( alg, data, labels):
    print "--- %s ---" % alg
    final_accuracy = 0
    params_final = [0.0, 0.0]
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(data, labels):
        new_data_train = data[train_index]
        new_data_test = data[test_index]
        new_labels_train = labels[train_index]
        new_labels_test = labels[test_index]
        acx = 0
        skf_intern = StratifiedKFold(n_splits=3)
        for intern_train_index, intern_test_index in skf_intern.split(new_data_train, new_labels_train):
            intern_data_train = new_data_train[intern_train_index]	
            intern_data_test = new_data_train[intern_test_index]	
            intern_labels_train = new_labels_train[intern_train_index]	
            intern_labels_test = new_labels_train[intern_test_index]
            params = get_intern_folds (alg, intern_data_train, intern_data_test, intern_labels_train, intern_labels_test)
            if params[0] > acx:
                acx = params[0]
                params_final[0] = params[1]
                if len(params) > 2:
                    params_final[1] = params[2]

        final_accuracy = final_accuracy + model_score(alg, params_final, 
                                                      new_data_train, 
                                                      new_labels_train, 
                                                      new_data_test, 
                                                      new_labels_test)
    final_accuracy = final_accuracy / 5
    print_results(alg, final_accuracy, params_final)

def model_score(alg, params, new_data_train, new_labels_train, new_data_test, new_labels_test):
    if 'svm' == alg:
        svm_model = SVM.SVC(C = params[0], gamma = params[1])
        svm_model.fit(new_data_train, new_labels_train)
        return svm_model.score(new_data_test, new_labels_test)
    elif 'knn' == alg:
        knn = KNeighborsClassifier(n_neighbors = params[0], n_jobs = 4)
        knn.fit(new_data_train, new_labels_train)
        return knn.score(new_data_test, new_labels_test)
    elif 'neural' == alg:
        clf = MLPClassifier(hidden_layer_sizes=(params[0],), solver='lbfgs')
        clf.fit(new_data_train, new_labels_train)
        return clf.score(new_data_test, new_labels_test)
    elif 'rf' == alg:
        clf = RandomForestClassifier (max_features = params[0], n_estimators = params[1])
        clf.fit(new_data_train, new_labels_train)
        return clf.score(new_data_test, new_labels_test)
    elif 'gbm' == alg:
        clf = GradientBoostingClassifier (learning_rate = params[0], n_estimators = params[1], max_depth = 5)
        clf.fit(new_data_train, new_labels_train)
        return clf.score(new_data_test, new_labels_test)

def get_intern_folds (alg, data_train, data_test, labels_train, labels_test):
    if 'svm' == alg:
        return svm_intern_folds(data_train, data_test, labels_train, labels_test)
    elif 'knn' == alg:
        return knn_intern_folds(data_train, data_test, labels_train, labels_test)
    elif 'neural' == alg:
        return neural_intern_folds(data_train, data_test, labels_train, labels_test)
    elif 'rf' == alg:
        return rf_intern_folds(data_train, data_test, labels_train, labels_test)
    elif 'gbm' == alg:
        return gbm_intern_folds(data_train, data_test, labels_train, labels_test)

def print_results(alg, final_accuracy, params):
    if 'svm' == alg:
        print("Acuracia:%s" % final_accuracy)
        print("Valor final hiperparametros (C=%s, Gamma=%s)" % (params[0], params[1]) )
    elif 'knn' == alg:
        print("Acuracia:%s" % final_accuracy)
        print("Valor final K (K=%s)" % (params[0]))
    elif 'neural' == alg:
        print("Acuracia:%s" % final_accuracy)
        print("Valor final parametros (Neurons=%s)" % (params[0]) )
    elif 'rf' == alg:
        print("Acuracia:%s" % final_accuracy)
        print("Valor final parametros (Feats=%s, Trees=%s)" % (params[0], params[1]) )
    elif 'gbm' == alg:
        print("Acuracia:%s" % final_accuracy)
        print("Valor final parametros (Learn Rate=%s, Trees=%s)" % (params[0], params[1]))


def main(argv=None):
    if argv is None:
        arv = sys.argv

    ## Data pre-processing    
    data = data_preprocess(datFileName)
    labels = getLabels(labelsFileName)
    labels = np.array(list(labels[:data.shape[0]]))

    ## kNN , PCA com 80% da variancia
    run_folds('knn', data, labels)

    ## SVM RBF 
    run_folds('svm', data, labels)

    ## Neural network
    run_folds('neural', data, labels)

    ## RF
    run_folds('rf', data, labels)

    ## GBM
    run_folds('gbm', data, labels)
    
if __name__ == "__main__":
    sys.exit(main())
