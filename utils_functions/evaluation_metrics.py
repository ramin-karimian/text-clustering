from sklearn.metrics import davies_bouldin_score , silhouette_score, classification_report
import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *
from nf1 import NF1
from clusim.clustering import Clustering, print_clustering
import clusim.sim

def NF1_communities_convertor(df,community_model):
    # comms = df[community_model].unique()
    df[community_model] = df[community_model].apply(str)
    comms = set([y for x in df[community_model].unique() for y in str(x).split(",")])
    new_comss = []
    for comm in comms:
        l = []
        for i in range(len(df)):
            if comm in df[community_model][i]:
                l.append(i)
        # l = list(df[df[community_model]==int(comm)].index)
        new_comss.append(l)
    return new_comss

def NF1_evaluation_metrics(df,community_model):
    y_true = NF1_communities_convertor(df,'class')
    y_pred = NF1_communities_convertor(df,community_model)
    nf = NF1(y_pred,y_true)
    results = nf.summary()
    f1_mean = round(results['details']['F1 mean'][0],3)
    nf1 = round(results['scores'].loc['NF1'][0],3)
    return f1_mean , nf1

def daviesBouldin(data,y_pred):
    score = davies_bouldin_score(data,y_pred)
    return score

def silhouette(data,y_pred):
    score = silhouette_score(data,y_pred)
    return score

def clusim_convertor(df,community_model):
    node2cid_pred = {}
    node2cid_true = {}
    for i in range(len(df)):
        # node2cid_true[i] = [df['class'][i]]
        # df['class'] = ['0']*23+['0,1']*5+['0,2']*5 + ['0,1']*5 + ['1']*22 + ['1,2']*5 + ['0,2']*5 + ['1,2']*5 + ['2']*22
        node2cid_true[i] = [int(eval(x)) for x in str(df['class'][i]).split(",")]
        node2cid_pred[i] = [int(eval(x)) for x in str(df[community_model][i]).split(",")]
    return node2cid_pred,node2cid_true

def element_centric(df,community_model):
    # c1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
    # c2 = Clustering(elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[2], 5:[2]})
    node2cid_pred,node2cid_true = clusim_convertor(df,community_model)
    c1 = Clustering(elm2clu_dict = node2cid_pred)
    c2 = Clustering(elm2clu_dict = node2cid_true)
    score = clusim.sim.element_sim(c1, c2, alpha = 0.9)
    return score

def f_measure(df,community_model):
    node2cid_pred,node2cid_true = clusim_convertor(df,community_model)
    c1 = Clustering(elm2clu_dict = node2cid_pred)
    c2 = Clustering(elm2clu_dict = node2cid_true)
    score = clusim.sim.fmeasure(c1, c2)
    return score


def jaccardIndex(df,community_model):
    node2cid_pred,node2cid_true = clusim_convertor(df,community_model)
    c1 = Clustering(elm2clu_dict = node2cid_pred)
    c2 = Clustering(elm2clu_dict = node2cid_true)
    score = clusim.sim.jaccard_index(c1, c2)
    return score


def randIndex(df,community_model):
    node2cid_pred,node2cid_true = clusim_convertor(df,community_model)
    c1 = Clustering(elm2clu_dict = node2cid_pred)
    c2 = Clustering(elm2clu_dict = node2cid_true)
    score = clusim.sim.rand_index(c1, c2)
    return score
