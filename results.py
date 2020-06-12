import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *

if __name__=="__main__":
    # dataset = "mixedOverlap"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = "cran-cisi-pubmed-100"
    dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-1000"
    numofclasses = 3
    dataversion = "V02"
    mode = ["exact3","more3"][0]
    netmodel = ['enn','knn'][0]
    evalMethods = ["element-centric","fmeasure","randIndex","f1-mean","jacardIndex"]
    models = [
            'Louvain_modularity','label_propagation','infomap',
            # 'k_clique',
            "asyn_fluidc",
            'kmeans','average_linkage',
            'lda','ptm',
            # 'nfm',
            'link_com_0.1cut','link_com_0.2cut','link_com_0.3cut',
            'link_com_0.4cut','link_com_0.5cut','link_com_0.6cut',
            'link_com_0.7cut','link_com_0.8cut','link_com_0.9cut'
    ]
    # representations = ['tf','tfidf','terms-tf','use','bert','elmo-weighted','elmo-default',
    #                    'LDA100','LDA50','LDA20','LDA3','PTM100','PTM50','PTM20','PTM3']
    representations = ['tf','tfidf','use','bert','elmo-weighted','elmo-default',
                       'LDA100','LDA50','LDA20','LDA3','PTM100','PTM50','PTM20','PTM3']

    datapath = f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}.xlsx"
    writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_results.xlsx",
                            engine='xlsxwriter')
    for rep in representations:
        if representations.index(rep)==0:
            df = pd.read_excel(datapath,sheet_name=rep)
            # applist = []
            # for model in models:
            #     if model not in df.keys():
            #         applist.append(model)
            #         applist.append(f'num_of_cl_{model}')
            #         applist.append(f"{model}_element-centric")
            #         applist.append(f"{model}_f1-mean")
            #         applist.append(f"{model}_fmeasure")
            #         applist.append(f"{model}_jacardIndex")
            #         applist.append(f"{model}_randIndex")
            # df = pd.DataFrame(df,columns = list(df.keys())+applist)
        else:
            df = pd.concat([df,pd.read_excel(datapath,sheet_name=rep)], join = 'outer').reset_index(drop=True)

    for evalMethod in evalMethods:
        df_new = pd.DataFrame(index = [model for model in models if "link_com" not in model] + ['link_comm'], columns = ['representation','threshold',
                                                         'similarity','num_of_nodes',
                                                         'num_of_edges','num_of_cl',
                                                         f"{evalMethod}" ])
        for model in models:
            if "link_com_" in model: continue
            if mode == "exact3": 
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]

            # tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                df_new.loc[model] = f"Non above {numofclasses} found"
                continue
            ind = tempdf.iloc[np.argmax(vals)].name
            for col in df_new.keys():
                if col == evalMethod:
                    df_new[col][model] = df[f"{model}_{evalMethod}"][ind]
                elif col == 'num_of_cl':
                    df_new[col][model] = df[f"{col}_{model}"][ind]
                else:
                    df_new[col][model] = df[col][ind]
        
        maxindval = 0
        for model in models:
            if "link_com_" not in model: continue
            if mode == "exact3": 
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]

            # tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                # df_new.loc[model] = f"Non above {numofclasses} found"
                continue
            
            ind = tempdf.iloc[np.argmax(vals)].name
            if df[f"{model}_{evalMethod}"][ind] > maxindval:
                maxindval = df[f"{model}_{evalMethod}"][ind] 
                for col in df_new.keys():

                    if col == evalMethod:
                        df_new[col]["link_comm"] = df[f"{model}_{evalMethod}"][ind]
                    elif col == 'num_of_cl':
                        df_new[col]["link_comm"] = df[f"{col}_{model}"][ind]
                    else:
                        df_new[col]["link_comm"] = df[col][ind]


        df_new.to_excel(writer, sheet_name=f"{evalMethod}")
    writer.save()
