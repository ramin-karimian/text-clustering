import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *
from sklearn.metrics import classification_report

def evaluation_metrics(df,reportdf,clusterlabel,start):
    y_true = df['class'].values.tolist()
    y_pred = df['Louvain_modularity'].apply(lambda x: clusterlabel[x]).values.tolist()
    rep = classification_report(y_true,y_pred,output_dict=True)
    for k in list(rep.keys())[-3:]:
        if k not in reportdf.keys(): reportdf[k]= None
        reportdf[k][0 +start] = rep[k]
    return reportdf

def update_report(datapath,reportdf,classDict):
    df = pd.read_excel(datapath)
    _,g = load_data(datapath[:-5]+".pkl")
    numofedges = len(g.edges)
    numofnodes = len(g.nodes)
    clusters = df['Louvain_modularity'].unique().tolist()
    start = len(reportdf)
    reportdf.loc[0+start] = None
    reportdf.loc[1+start] = None
    name = datapath.split('\\')[-1].split('.xlsx')[0].split("_")
    reportdf.loc[0+start]['model'] = f"{name[4]}_{name[6]}_{name[7]}_{name[0]}_{name[3]}"
    reportdf.loc[0+start][f'num_of_cl'] = len(clusters)
    reportdf.loc[0+start][f'num_of_edges'] = numofedges
    reportdf.loc[0+start][f'num_of_nodes'] = numofnodes
    clusterlabel = [None]*5
    for cl in clusters:
        counts = [0,0,0,0,0]
        percentages = [0,0,0,0,0]
        if f'cl_{cl+1}' not in reportdf.keys():reportdf[f'cl_{cl+1}'] = None
        # class_report = df[df['Louvain_modularity']==cl]['class'].value_counts()
        class_label__report = df[df['Louvain_modularity']==cl]['class_label'].value_counts()
        for x in class_label__report.keys():
            counts[classDict[x]] = class_label__report[x]
            percentages[classDict[x]] = round(class_label__report[x] / sum(class_label__report.values),2)
        reportdf.loc[0+start][f'cl_{cl+1}'] = counts
        reportdf.loc[1+start][f'cl_{cl+1}'] = percentages
        if len(clusters)==len(classDict):
            clusterlabel[cl] = np.argmax(counts)
    if len(clusters)==len(classDict): reportdf = evaluation_metrics(df,reportdf,clusterlabel,start)
    return reportdf

if __name__=="__main__":
    # datapath = f"../output/bbc_preprocessed_data_V01_use_network_th_0.5.xlsx"
    reportpath = f"../output/bbc_report.xlsx"
    classDict = {"business":0,"entertainment":1,"politics":2,"sport":3,"tech":4}
    path = f'../output'
    reportdf = pd.DataFrame(columns=['model','num_of_cl','num_of_nodes','num_of_edges','accuracy', 'macro avg', 'weighted avg'])
    #
    # if reportpath.split("/")[-1] not in os.listdir(f"../output"):
    #     reportdf = pd.DataFrame(columns=['model','num_of_cl','accuracy', 'macro avg', 'weighted avg'])
    # else:
    #     reportdf = pd.read_excel(reportpath)

    for filename in  os.listdir(path):
        if not filename.endswith('.xlsx'): continue
        if filename.endswith("report.xlsx") : continue
        datapath = os.path.join(path,filename)
        reportdf = update_report(datapath,reportdf,classDict)

    reportdf.to_excel(reportpath,index=None)
