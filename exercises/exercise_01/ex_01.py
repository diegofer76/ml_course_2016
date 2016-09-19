#!/usr/bin/python

import sys,os,csv
import pandas
import numpy
import numpy as np
from sklearn.decomposition import PCA
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
    print(rawData.shape)
    return rawData.values

def applyPCA(rawData):
    print "Apply PCA"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    
    data = np.array(rawData[:][0:colNum - 1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    #pca = PCA(colNum)
    pca = PCA(n_components=80)
    pca.fit(data)
    #PCA(copy=True, n_components=colNum, whiten=True)
    print "params" ,  pca.get_params()
    cov_mat = pca.get_covariance()
    print cov_mat.shape
    #print(pca.explained_variance_ratio_)
    
    matrix = data
    matrix = np.matrix(matrix) * np.matrix(matrix).transpose() 
    U,S,V = np.linalg.svd(matrix) 
    #T = U * S
    print U.shape, S.shape, V.shape

    s_sum_all = sum(S)
    for i in range(len(S)):
        if sum(S[0:i]) / s_sum_all >= 0.8 :
            print i, " --- ", sum(S[0:i]) / s_sum_all
            break


def chooseComponentsNumber():
    # Try PCA with 
    for k in range(colNum):
        print "" 
        # Compute U, z_1, x_aprox
        # Check desigualdade

def meanNormalization(rawdata):
    for i in range(rawdata.shape[0]):
        np.mean(data[i])
    return data

def main(argv=None):
    if argv is None:
        arv = sys.argv
    data = loadCsvData(dirPath + "/" + datFileName)
    applyPCA(data)

if __name__ == "__main__":
    sys.exit(main())
