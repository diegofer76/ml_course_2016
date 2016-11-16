#!/usr/bin/python

# ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys,os,csv
import pandas
import numpy as np
import math
import string
import random
import time
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

import urllib2
import pandas as pd

datFileName="train.csv"
datFileNameTest="test.csv"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]
alphabet = (list(string.lowercase) + list(string.uppercase))
vec = DictVectorizer()


def load_data(fileName):
    raw_data = open(fileName, 'rb')
    rawData = pandas.read_csv(raw_data, delimiter=",", header=None)
    return rawData

def data_preprocessing(datFileName):
    orig_data = load_data(datFileName)
    orig_data.columns = [ letter for letter in alphabet][0:33]
    adjust_data = np.array(orig_data[alphabet[0]])

    data = orig_data[alphabet[1:33]]
    data = data.T.to_dict().values()
    data = vec.fit_transform(data).toarray()
    return data, adjust_data

def data_preprocessing_test(datFileName):
    data = load_data(datFileName)
    orig_data = load_data(datFileName)
    orig_data.columns = [ letter for letter in alphabet][1:33]
    data = orig_data.T.to_dict().values()
    data = vec.transform(data).toarray()
    return data

def changeCharNum(data):
    for line in data:
    #for line in data.transpose():
        #print "->", np.unique(line)
        for index in range(len(line)):
            if line[index] in charDigits:
                line[index] = charDigits[line[index]]
    return data

def svr_check(data, adjust_data):
    #extern stratified kfold
    skf_ext = StratifiedKFold(n_splits=5)
    #Support vector machine
    svr_params = [{'kernel': ['rbf'], 'gamma': [2**-15, 2**-10, 2**-5, 2**(0), 2**5], 
                                      'C': [2**-5, 2**0, 2**5, 2**10]}]
    accuracies_svr = []
    svr_regressor = None
    fold = 0
    max_mae = -1

    for train_index, test_index in skf_ext.split(data, adjust_data):
        start_time = time.time()
        print "SVR iteration Fold %d " % fold
        fold += 1
         #train and test data
        train_data, test_data = data[train_index], data[test_index]

         #train and test labels
        train_adjust_data, test_adjust_data = adjust_data[train_index], adjust_data[test_index]

        svr = GridSearchCV(SVR(), svr_params, n_jobs=4)
        svr.fit(train_data, train_adjust_data)
        mae = mean_absolute_error(test_adjust_data, svr.predict(test_data))

        if mae > max_mae:
            max_mae = mae
            svr_regressor = svr

        print("MAE: %.4f, MAX_MAE:  %4f" % (mae, max_mae))
        elapsed_time = (time.time() - start_time) / 60
        print "--- Elapsed time:", elapsed_time , " minutes"

    return svr_regressor

def gbr_check(data, adjust_data):
    #extern stratified kfold
    skf_ext = StratifiedKFold(n_splits=5)

    #Gradient Boosting Regressor
    gbr_params = {'max_depth': [4, 6], 'min_samples_leaf': [3, 5, 9, 17],
          'learning_rate': [0.1, 0.01], 'loss': ['ls']}

    accuracies_gbr = []
    gbr_regressor = None
    fold = 0
    max_mae = -1

    for train_index, test_index in skf_ext.split(data, adjust_data):
        print "GBR iteration Fold %d " % fold
        fold += 1
         #train and test data
        train_data, test_data = data[train_index], data[test_index]

        #train and test labels
        train_adjust_data, test_adjust_data = adjust_data[train_index], adjust_data[test_index]
        est = GradientBoostingRegressor(n_estimators=500)
        gbr = GridSearchCV (est, gbr_params, n_jobs=4)
        gbr.fit(train_data, train_adjust_data)
        mae = mean_absolute_error(test_adjust_data, gbr.predict(test_data))

        if mae > max_mae:
            max_mae = mae
            gbr_regressor = gbr

        print("MAE: %.4f, MAX_MAE:  %.4f" % (mae, max_mae))

    return gbr_regressor

def rf_check(data, adjust_data):
    #extern stratified kfold
    skf_ext = StratifiedKFold(n_splits=5)
    rf_params = { "n_estimators"      : [100, 250, 300, 400],
                   "max_features"      : [3, 5],
                   "max_depth"         : [10, 20],
                   "min_samples_split" : [2, 4] ,
                   "bootstrap": [True, False]}

    rf_regressor = None
    fold = 0
    max_mae = -1

    for train_index, test_index in skf_ext.split(data, adjust_data):
        print "RF iteration Fold %d " % fold
        fold += 1
         #train and test data
        train_data, test_data = data[train_index], data[test_index]

        #train and test labels
        train_adjust_data, test_adjust_data = adjust_data[train_index], adjust_data[test_index]
        rfr = RandomForestRegressor(random_state=30)
        rf = GridSearchCV (rfr, rf_params, n_jobs=4)
        rf.fit(train_data, train_adjust_data)
        mae = mean_absolute_error(test_adjust_data, rf.predict(test_data))

        if mae > max_mae:
            max_mae = mae
            rf_regressor = rf

        print("MAE: %.4f, MAX_MAE:  %.4f" % (mae, max_mae))

    return rf_regressor

def main(argv=None):
    if argv is None:
        arv = sys.argv

    ## Data pre-processing    
    train_data, train_adjust_data = data_preprocessing(datFileName)
    test_data = data_preprocessing_test(datFileNameTest)

    start_time = time.time()

    # principal component analysis
    #pca = PCA(n_components=0.999).fit(train_data)
    pca = PCA(n_components=0.99).fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    print train_data.shape
    print test_data.shape

    # Support vector machine
    print "---- SVR ----"
    svr = svr_check(train_data, train_adjust_data)
    print "SVR predict:", svr.predict(test_data)

    #Gradient Boosting Regressor
    print "---- GBR ----"
    gbr = gbr_check(train_data, train_adjust_data)
    print "gbr predict:", gbr.predict(test_data)

    #Random Forest Regressor
    print "---- RF ----"
    rf = rf_check(train_data, train_adjust_data)
    print "rf predict:", rf.predict(test_data)

    elapsed_time = time.time() - start_time
    print "Total elapsed time:", elapsed_time

if __name__ == "__main__":
    sys.exit(main())
