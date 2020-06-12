import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from time import time as tm
import numpy as np
from sklearn.decomposition import NMF

ward = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
complete = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
average = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
single = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')


def convert_output(modelname,output):
    partition = {}
    if modelname in ["lda","ptm","nfm"]:
        for i in range(len(output)):
            partition[i] = np.argmax(output[i])
    elif modelname in ["average_linkage","kmeans"]:
        for i in range(len(output)):
            partition[i] = output[i]
    return partition


def average_linkage(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    average = AgglomerativeClustering(n_clusters=num_cl, affinity=distfunc, linkage='average')
    average_predict = average.fit_predict(data)
    if return_dict is not None:
        for i in range(len(average_predict)):
            return_dict[i]["average_linkage"] = average_predict[i]
        print("average_linkage",f" took {tm()-t1} s")
        return  return_dict
    else:
        partition = convert_output("average_linkage",average_predict)
        return partition

def nmf(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    distros = NMF(n_components=num_cl, random_state=1).fit_transform(data)
    if return_dict is not None:
        for i in range(len(distros)):
            return_dict[i]["nmf"] = np.argmax(distros[i])
        print("nmf",f" took {tm()-t1} s")
        return  return_dict
    else:
        partition = convert_output("nmf",distros)
        return partition

def lda(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    if return_dict is not None:
        for i in range(len(data)):
            return_dict[i]["lda"] = np.argmax(data[i])
        print("lda",f" took {tm()-t1} s")
        return  return_dict
    else:
        partition = convert_output("lda",data)
        return partition

def ptm(data,return_dict,num_cl,distfunc = 'euclidean'):
    t1=tm()
    if return_dict is not None:
        for i in range(len(data)):
            return_dict[i]["ptm"] = np.argmax(data[i])
        print("ptm",f" took {tm()-t1} s")
        return  return_dict
    else:
        partition = convert_output("ptm",data)
        return partition

def kmeans(data,return_dict,num_cl,distfunc = 'euclidean'):
    # kmeans = KMeans(n_clusters=num_cl, random_state=0).fit(X)
    t1=tm()
    partitions = KMeans(n_clusters=num_cl).fit(data)
    partitions = partitions.labels_
    if return_dict is not None:
        for i in range(len(partitions)):
            return_dict[i]["kmeans"] = partitions[i]
        print("kmeans",f" took {tm()-t1} s")
        return  return_dict
    else:
        partition = convert_output("kmeans",partitions)
        return partition
