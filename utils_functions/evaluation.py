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
from multiprocessing import Process, Manager
from utils_functions.evaluation_metrics import daviesBouldin, silhouette,\
    NF1_evaluation_metrics, element_centric,f_measure,jaccardIndex,randIndex


def evaluation_metrics(df,reportdf,clusterlabel,start,community_model):
    # y_true = df['class'].values.tolist()
    # y_pred = df[community_model].apply(lambda x: clusterlabel[x]).values.tolist()

    # rep = classification_report(y_true,y_pred,output_dict=True)
    f1_mean , nf1 = NF1_evaluation_metrics(df,community_model)
    elecent = element_centric(df,community_model)
    fscore = f_measure(df,community_model)
    jscore = jaccardIndex(df,community_model)
    rscore = randIndex(df,community_model)

    # if f"{community_model}_nf1" not in reportdf.keys(): reportdf[f"{community_model}_nf1"]= None
    # reportdf[f"{community_model}_accuracy"][0 +start] = rep['accuracy']
    # reportdf[f"{community_model}_f1score"][0 +start] = round((rep['weighted avg']['f1-score']),3)
    reportdf[f"{community_model}_f1-mean"][0 +start] = f1_mean
    reportdf[f"{community_model}_fmeasure"][0 +start] = fscore
    reportdf[f"{community_model}_element-centric"][0 +start] = elecent
    reportdf[f"{community_model}_jacardIndex"][0 +start] = jscore
    reportdf[f"{community_model}_randIndex"][0 +start] = rscore
    # reportdf[f"{community_model}_nf1"][0 +start] = nf1
    return reportdf

def update_report(datapath,reportdf,classDict,community_models):
    df = pd.read_excel(datapath)
    _,g = load_data(datapath[:-5]+".pkl")
    numofedges = len(g.edges)
    numofnodes = len(g.nodes)
    start = len(reportdf)
    reportdf.loc[start] = None
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
        if f'num_of_cl_{community_model}' not in reportdf.keys():reportdf[f'num_of_cl_{community_model}'] = None
        # if f"accuracy_{community_model}" not in reportdf.keys(): reportdf[f"accuracy_{community_model}"]= None
        # if f"{community_model}_f1score" not in reportdf.keys(): reportdf[f"{community_model}_f1score"]= None
        if f"{community_model}_element-centric" not in reportdf.keys(): reportdf[f"{community_model}_element-centric"]= None
        if f"{community_model}_f1-mean" not in reportdf.keys(): reportdf[f"{community_model}_f1-mean"]= None
        if f"{community_model}_fmeasure" not in reportdf.keys(): reportdf[f"{community_model}_fmeasure"]= None
        if f"{community_model}_jacardIndex" not in reportdf.keys(): reportdf[f"{community_model}_jacardIndex"]= None
        if f"{community_model}_randIndex" not in reportdf.keys(): reportdf[f"{community_model}_randIndex"]= None

        try:
            clusters = set([y for x in df[community_model].unique() for y in str(x).split(",")])
        except:
            if "link_com_" not in community_model:
                if community_model not in ['ptm','lda','nfm','asyn_fluidc']:
                    print(f"error in {community_model}")
            continue
        clusterlabel = [None]*len(clusters)

        reportdf.loc[start][f'num_of_cl_{community_model}'] = len(clusters)
        try:
            reportdf = evaluation_metrics(df,reportdf,clusterlabel,start,community_model)
        except:
            print(f"{community_model} _ {datapath} ")
            # break
    return reportdf

def seperate_sheets(reportdf,writer):
    for rep in reportdf['representation'].unique().tolist():
        df = reportdf[reportdf['representation'] == rep]
        df.to_excel(writer,index=None,sheet_name=f"{rep}")
    writer.save()

def evaluate(reportpath,path,community_models,classDict):
    reportdf = pd.DataFrame(columns=['representation','threshold','similarity',
                                     'num_of_nodes','num_of_edges','daviesBouldin',
                                     'silhouette'])
    writer = pd.ExcelWriter(reportpath,engine='xlsxwriter')
    for filename in  os.listdir(path):
        if not filename.endswith('.xlsx'): continue
        if filename.endswith("report.xlsx") : continue
        if filename == "linked_community_res.xlsx" : continue
        if "cosineSimilarity" not in filename: continue
        datapath = os.path.join(path,filename)
        name = datapath.split('\\')[-1].split('.xlsx')[0].split("_")
        if name[6]!=networkmodel: continue
        print(filename)
        reportdf = update_report(datapath,reportdf,classDict[dataset],community_models)
    seperate_sheets(reportdf,writer)

def multiprocessfunc(core,start,end,filenames,reportdf,classDict,community_models,path,return_dict):
    for i in range(start,end):
        if (i-start) %1==0:
            print(f"i = {i-start}/{end-start} corenum {core}")
        filename = filenames[i]
        datapath = os.path.join(path,filename)
        reportdf = update_report(datapath,reportdf,classDict,community_models)
    return_dict[core]=reportdf

def multiprocess_evaluation(reportpath,path,community_models,classDict):
    reportdf = pd.DataFrame(columns=['representation','threshold','similarity',
                                     'num_of_nodes','num_of_edges','daviesBouldin',
                                     'silhouette'])
    writer = pd.ExcelWriter(reportpath,engine='xlsxwriter')
    filenames = []
    for filename in  os.listdir(path):
        if not filename.endswith('.xlsx'): continue
        if filename.endswith("report.xlsx") : continue
        if filename == "linked_community_res.xlsx" : continue
        if "cosineSimilarity" not in filename: continue
        datapath = os.path.join(path,filename)
        name = datapath.split('\\')[-1].split('.xlsx')[0].split("_")
        if name[6]!=networkmodel: continue
        filenames.append(filename)

    end = 0
    step = int(len(filenames)/cores)
    return_dict  = Manager().dict()
    processess = []
    for i in range(cores):
        start = end
        if i ==cores-1:
            end = len(filenames)
        else:
            end = start + step
        p = Process(target = multiprocessfunc,args = (i+1,start,end,filenames,reportdf,classDict,community_models,path,return_dict))
        p.start()
        processess.append(p)

    for p in processess:
        p.join()

    reportdf = return_dict[1]
    for core in range(2,cores+1):
        reportdf = pd.concat([reportdf,return_dict[core]], join = 'outer').reset_index(drop=True)
    seperate_sheets(reportdf,writer)

if __name__=="__main__":
    # dataset = 'cran-cisi-pubmed'
    # dataset = "cran-cisi-pubmed-1000"
    dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    # dataset = "cran-cisi-pubmed-100"
    # dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = 'mixed'
    # dataset = 'bbcsport-small'
    # dataset = 'mixed'
    # dataset = 'bbc'
    # dataset = 'twitter-cor2'
    # dataset = 'twitter'
    # dataset = 'mixedOverlap'
    data_version = "V02"
    # data_version = "V01"
    networkmodel = ['enn','knn'][0]
    cores = 15
    community_models = ['Louvain_modularity','label_propagation','infomap',
                        # 'k_clique',
                        "asyn_fluidc",'kmeans','average_linkage',
                        'lda','ptm',
                        # 'nfm',
                        'link_com_0.1cut','link_com_0.2cut','link_com_0.3cut',
                        'link_com_0.4cut','link_com_0.5cut','link_com_0.6cut',
                        'link_com_0.7cut','link_com_0.8cut','link_com_0.9cut',
                        # 'link_com_0.1cut','link_com_0.3cut','link_com_0.5cut','link_com_0.7cut'
                        ]
    classDict ={
        'bbc':{"business":0,"entertainment":1,"politics":2,"sport":3,"tech":4},
        'bbcsport':{"athletics":0,"cricket":1,"football":2,"rugby":3,"trennis":4},
        'cran_cisi_pubmed':{"cran":0,"cisi":1,"pubmed":2},
        'cran-cisi-pubmed-300-overlapped':{"cran":0,"cisi":1,"pubmed":2},
        'cran-cisi-pubmed-1000-overlapped':{"cran":0,"cisi":1,"pubmed":2},
        'cran-cisi-pubmed-300':{"cran":0,"cisi":1,"pubmed":2},
        'cran-cisi-pubmed-100-overlapped':{"cran":0,"cisi":1,"pubmed":2},
        'cran-cisi-pubmed-100':{"cran":0,"cisi":1,"pubmed":2},
        'mixed':{"cran":0,"cisi":1,"pubmed":2},
        'mixedOverlap':{"cran":0,"cisi":1,"pubmed":2},
        'twitter':{"art":0,"business":1,"education":2,"food":3,"technology":4},
        'twitter-cor':{"art , business":0, "art , technology":1,"business , technology":2,"education , art":3,"education , business":4,"education , technology":5,"food , art":6,"food , business":7,"food , education":8,"food , technology":9},
        'twitter-cor2':{"business , technology":0,"business":1,"food , business":2,"food , technology":3,"food":4,"technology":5}
        # 'twitter-cor2':{"business":0,"food":1,"technology":2}
    }
    reportpath = f"../data/output/{dataset}/{data_version}/{dataset}_{data_version}_report_{networkmodel}.xlsx"
    path = f'../data/output/{dataset}/{data_version}/network'
    multiprocess_evaluation(reportpath,path,community_models,classDict)
