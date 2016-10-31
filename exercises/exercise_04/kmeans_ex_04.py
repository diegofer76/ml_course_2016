#!/usr/bin/python

import sys,os,csv
import pandas
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

datFileName="cluster-data.csv"
labelsFileName="cluster-data-class.csv"
dirPath=os.path.dirname(os.path.realpath(__file__))
classList=[]
data=[]

def load_data(fileName):
    raw_data = open(fileName, 'rb')
    #rawData = pandas.read_csv(raw_data, delimiter=" ")
    rawData = pandas.read_csv(raw_data, delimiter=",", skiprows=1)
    return rawData.values

def getData(rawData):
    #print "\n---- Getting data from File ----"
    lineNum = rawData.shape[0]
    colNum = rawData.shape[1]
    data = np.array(rawData[0:lineNum, 0:colNum-1])
    for i in range(lineNum):
        classList.append(rawData[i][colNum - 1])
    return [data, np.array(classList) ]

def get_labels(fileName):
    labelData = load_data(dirPath + "/" + fileName)
    labels = labelData[:,0].clip(min=0)
    return np.array(labels)


def bench_k_means_inter(estimator, name, data):
    estimator.fit(data)
    best_score = 0.0
    best_metric = ''
    scores = [metrics.silhouette_score(data, estimator.labels_,metric='euclidean', sample_size=5416),
             metrics.calinski_harabaz_score(data, estimator.labels_)/10000 ]
    print('% 9s   %.3f   %.3f'
          % (name, scores[0], scores[1]))

    if scores[0] > best_score:          
        best_score = scores[0]
        best_metric = "Silouette"
    
    if scores[1] > best_score:          
        best_score = scores[1]
        best_metric = "calinski"

    return [best_score, best_metric]

def bench_k_means_ext(estimator, name, data, labels):
    estimator.fit(data) 
    best_score = 0.0
    best_metric = ''
    scores = [metrics.homogeneity_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_) ]
    print('% 9s   %.3f   %.3f   %.3f   %.3f'
          % (name, scores[0] , scores[1], scores[2], scores[3]) ) 

    if scores[0] > best_score:          
        best_score = scores[0]
        best_metric = "Homogeneity"
    
    if scores[1] > best_score:          
        best_score = scores[1]
        best_metric = "v_measure"

    if scores[2] > best_score:          
        best_score = scores[2]
        best_metric = "adjusted_rand"

    if scores[3] > best_score:          
        best_score = scores[3]
        best_metric = "adjusted_mutual"

    return [best_score, best_metric]

def main(argv=None):
    if argv is None:
        arv = sys.argv

    data = load_data(datFileName)
    labels = get_labels(labelsFileName)
    data = scale(data)

    best_metric_int = ''
    best_metric_ext = ''
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("--- Metricas Internas ---")
    print('%9s   %s  %s' % ('K', 'Silhouette', 'Calinski'))
    best_score = 0.0
    best_metric = ''
    best_k = 0
    for n_clusters in range_n_clusters:
        new_name = "k=" + str(n_clusters)
        [last_best_score, last_best_metric] = bench_k_means_inter(KMeans(init='k-means++', n_clusters=n_clusters, n_init=5), name=new_name, data=data)
        if last_best_score > best_score:
            best_score = last_best_score
            best_metric_int = last_best_metric
            best_k = n_clusters

    print "best score: ", best_score
    print "best metric intern: ", best_metric_int
    print "best k:", best_k

    best_score=0.0
    print("")
    print("--- Metricas Externas ---")
    print('%9s   %s  %s %s  %s' % ('K', 'Homogen', 'v_meas', 'adj_rand', 'mutual'))
    for n_clusters in range_n_clusters:
        new_name = "k=" + str(n_clusters)
        [last_best_score, last_best_metric] = bench_k_means_ext(KMeans(init='k-means++', n_clusters=n_clusters, n_init=5), name=new_name, data=data, labels=labels)
        if last_best_score > best_score:
            best_score = last_best_score
            best_metric_ext = last_best_metric
            best_k = n_clusters

    print "best score: ", best_score
    print "best metric extern: ", best_metric_ext
    print "best k: ", best_k

    final_score_int=[]   
    final_score_ext=[]   

    for n_clusters in range_n_clusters:
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=5)
        kmeans.fit(data)
        final_score_int.append( metrics.silhouette_score(data, kmeans.labels_, metric='euclidean', sample_size=5416))
        final_score_ext.append( metrics.homogeneity_score(labels, kmeans.labels_))

    legen_int = 'Score Interna - ' + str(best_metric_int)
    legen_ext = 'Score Externa - ' + str(best_metric_ext)

    plt.plot(range_n_clusters, final_score_int, 'bs--', linewidth=4, markersize=10, label=legen_int)
    plt.plot(range_n_clusters, final_score_ext, 'r^:', linewidth=4, markersize=10, label=legen_ext)
    plt.xlabel('Numero de clusters (K)')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
    
