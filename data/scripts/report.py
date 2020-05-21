import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *
from sklearn.metrics import classification_report
import networkx as nx
from nf1 import NF1
from main import load_data_forsim
from utils_functions.evaluation_metrics import daviesBouldin,silhouette

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



def evaluation_metrics(df,reportdf,clusterlabel,start,community_model):
    y_true = df['class'].values.tolist()
    y_pred = df[community_model].apply(lambda x: clusterlabel[x]).values.tolist()

    rep = classification_report(y_true,y_pred,output_dict=True)
    f1_mean , nf1 = NF1_evaluation_metrics(df,community_model)


    # if f"accuracy_{community_model}" not in reportdf.keys(): reportdf[f"accuracy_{community_model}"]= None
    if f"{community_model}_f1score" not in reportdf.keys(): reportdf[f"{community_model}_f1score"]= None
    if f"{community_model}_f1-mean" not in reportdf.keys(): reportdf[f"{community_model}_f1-mean"]= None
    if f"{community_model}_nf1" not in reportdf.keys(): reportdf[f"{community_model}_nf1"]= None
    # reportdf[f"{community_model}_accuracy"][0 +start] = rep['accuracy']
    reportdf[f"{community_model}_f1score"][0 +start] = round((rep['weighted avg']['f1-score']),3)
    reportdf[f"{community_model}_f1-mean"][0 +start] = f1_mean
    reportdf[f"{community_model}_nf1"][0 +start] = nf1
    return reportdf

def update_report(datapath,reportdf,classDict,community_models):
    df = pd.read_excel(datapath)
    _,g = load_data(datapath[:-5]+".pkl")
    numofedges = len(g.edges)
    numofnodes = len(g.nodes)
    start = len(reportdf)
    reportdf.loc[start] = None
    # reportdf.loc[1+start] = None
    name = datapath.split('\\')[-1].split('.xlsx')[0].split("_")

    reportdf.loc[start][f'representation'] = name[4]
    reportdf.loc[start][f'threshold'] = name[7]
    reportdf.loc[start][f'similarity'] = name[8]
    reportdf.loc[start][f'num_of_nodes'] = numofnodes
    reportdf.loc[start][f'num_of_edges'] = numofedges
    if name[4] not in ['terms-tf']:
        repre_datapath = os.path.join(datapath,"../../representation","_".join(name[:5])+f".pkl")
        repre_data =load_data_forsim(repre_datapath,name=name[4])
        daviesBouldin_score = daviesBouldin(repre_data,df['class'].values.tolist())
        silhouette_score = silhouette(repre_data,df['class'].values.tolist())
        reportdf[f"silhouette"][0 +start] = silhouette_score
        reportdf[f"daviesBouldin"][0 +start] = daviesBouldin_score

    for community_model in community_models:
        try:
            clusters = df[community_model].unique().tolist()
        except:
            print(f"error in {community_model}")
            continue
        clusterlabel = [None]*len(clusters)
        if f'num_of_cl_{community_model}' not in reportdf.keys():reportdf[f'num_of_cl_{community_model}'] = None
        reportdf.loc[start][f'num_of_cl_{community_model}'] = len(clusters)
        for cl in clusters:
            counts = [0]*len(classDict)
            percentages = [0.0]*len(classDict)
            class_label__report = df[df[community_model]==cl]['class_label'].value_counts()
            for x in class_label__report.keys():
                counts[classDict[x]] = class_label__report[x]
                percentages[classDict[x]] = round(class_label__report[x] / sum(class_label__report.values),2)
            clusterlabel[cl] = np.argmax(counts)
        reportdf = evaluation_metrics(df,reportdf,clusterlabel,start,community_model)
    return reportdf

def seperate_sheets(reportdf,writer):
    for rep in reportdf['representation'].unique().tolist():
        df = reportdf[reportdf['representation'] == rep]
        df.to_excel(writer,index=None,sheet_name=f"{rep}")
    writer.save()


if __name__=="__main__":
    # dataset = 'cran-cisi-pubmed'
    dataset = 'bbcsport'
    # dataset = 'bbcsport-small'
    # dataset = 'mixed'
    # dataset = 'bbc'
    # dataset = 'twitter-cor2'
    data_version = "V02"
    # data_version = "V01"
    networkmodel = ['enn','knn'][0]
    community_models = ['Louvain_modularity','label_propagation','infomap',
                        # 'k_clique',
                        "asyn_fluidc",'kmeans','average_linkage']
    classDict ={
        'bbc':{"business":0,"entertainment":1,"politics":2,"sport":3,"tech":4},
        'bbcsport':{"athletics":0,"cricket":1,"football":2,"rugby":3,"tennis":4},
        'cran_cisi_pubmed':{"cran":0,"cisi":1,"pubmed":2},
        'mixed':{"cran":0,"cisi":1,"pubmed":2},
        'twitter':{"art":0,"business":1,"education":2,"food":3,"technology":4},
        'twitter-cor':{"art , business":0, "art , technology":1,"business , technology":2,"education , art":3,"education , business":4,"education , technology":5,"food , art":6,"food , business":7,"food , education":8,"food , technology":9},
        'twitter-cor2':{"business , technology":0,"business":1,"food , business":2,"food , technology":3,"food":4,"technology":5}
        # 'twitter-cor2':{"business":0,"food":1,"technology":2}
    }
    reportpath = f"../output/{dataset}/{data_version}/{dataset}_{data_version}_report_{networkmodel}.xlsx"
    path = f'../output/{dataset}/{data_version}/network'
    # reportdf = pd.DataFrame(columns=['model','num_of_nodes','num_of_edges'])
    reportdf = pd.DataFrame(columns=['representation','threshold','similarity',
                                     'num_of_nodes','num_of_edges','daviesBouldin',
                                     'silhouette'])
    writer = pd.ExcelWriter(reportpath,engine='xlsxwriter')

    for filename in  os.listdir(path):
        if not filename.endswith('.xlsx'): continue
        if filename.endswith("report.xlsx") : continue
        if filename == "linked_community_res.xlsx" : continue
        datapath = os.path.join(path,filename)
        name = datapath.split('\\')[-1].split('.xlsx')[0].split("_")
        if name[6]!=networkmodel: continue
        print(filename)
        reportdf = update_report(datapath,reportdf,classDict[dataset],community_models)

    seperate_sheets(reportdf,writer)
    # reportdf.to_excel(reportpath,index=None)
