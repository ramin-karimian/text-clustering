import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from time import time as tm
import numpy as np
from sklearn.decomposition import NMF

ward = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
complete = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
average = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
single = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')

def average_linkage(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    average = AgglomerativeClustering(n_clusters=num_cl, affinity=distfunc, linkage='average')
    average_predict = average.fit_predict(data)
    for i in range(len(average_predict)):
        return_dict[i]["average_linkage"] = average_predict[i]
    print("average_linkage",f" took {tm()-t1} s")
    return  return_dict

def nmf(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    distros = NMF(n_components=num_cl, random_state=1).fit_transform(data)
    for i in range(len(distros)):
        return_dict[i]["nmf"] = np.argmax(distros[i])
    print("nmf",f" took {tm()-t1} s")
    return  return_dict

def lda(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    for i in range(len(data)):
        return_dict[i]["lda"] = np.argmax(data[i])
    print("lda",f" took {tm()-t1} s")
    return  return_dict

def kmeans(data,return_dict,num_cl,distfunc = 'euclidean'):
    # kmeans = KMeans(n_clusters=num_cl, random_state=0).fit(X)
    t1=tm()
    partitions = KMeans(n_clusters=num_cl).fit(data)
    partitions = partitions.labels_
    for i in range(len(partitions)):
        return_dict[i]["kmeans"] = partitions[i]
    print("kmeans",f" took {tm()-t1} s")
    return  return_dict
