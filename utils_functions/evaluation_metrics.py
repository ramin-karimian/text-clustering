from sklearn.metrics import davies_bouldin_score , silhouette_score, classification_report
import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *
from nf1 import NF1

def NF1_communities_convertor(df,community_model):
    comms = df[community_model].unique()
    new_comss = []
    for comm in comms:
        l = list(df[df[community_model]==comm].index)
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

