import pandas as pd
import numpy as np
import re
import os
import pickle
from data.scripts.utils import *

def eachrep_eachmodel(dataset,dataversion,datapath,netmodel,mode,models,representations):
    # writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_results.xlsx", engine='xlsxwriter')
    for rep in representations:
        df = pd.read_excel(datapath,sheet_name=rep)
        writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_{rep}.xlsx",
                            engine='xlsxwriter')

        for model in models:
            # print(model)
            if "link_com_" in model: continue
            if mode == "exact3":
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]

            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                print(f"Non above {numofclasses} found for {model}")
                continue
            df_new = pd.DataFrame(index = tempdf.index , columns = ['threshold','similarity','num_of_nodes',
                         'num_of_edges','num_of_cl',f"{evalMethod}" ])
            for ind in list(tempdf.index):
                for col in df_new.keys():
                    if col == evalMethod:
                        df_new.loc[ind][col] = df[f"{model}_{evalMethod}"][ind]
                    elif col == 'num_of_cl':
                        df_new.loc[ind][col] = df[f"{col}_{model}"][ind]
                    else:
                        df_new.loc[ind][col] = df[col][ind]
            df_new.to_excel(writer,sheet_name=f"{model}")

        for model in models:
            if "link_com_" not in model: continue
            if mode == "exact3":
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                print(f"Non above {numofclasses} found for {model}")
                continue
            ind = tempdf.iloc[np.argmax(vals)].name
            df_new = pd.DataFrame(index = tempdf.index , columns = ['threshold','similarity','num_of_nodes',
                     'num_of_edges','num_of_cl',f"{evalMethod}" ])
            for ind in list(tempdf.index):
                for col in df_new.keys():
                    if col == evalMethod:
                        df_new.loc[ind][col] = df[f"{model}_{evalMethod}"][ind]
                    elif col == 'num_of_cl':
                        df_new.loc[ind][col] = df[f"{col}_{model}"][ind]
                    else:
                        df_new.loc[ind][col] = df[col][ind]
            df_new.to_excel(writer,sheet_name=f"{model}")
        writer.save()

def bestofreps_for_models(datapath,mode,models,representations):
    writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_besfofreps-for-models.xlsx",
                            engine='xlsxwriter')
    for model in models:
        if "link_com_" in model: continue
        df_new = pd.DataFrame(index = representations, columns = ['representation','threshold','similarity','num_of_nodes',
                         'num_of_edges','num_of_cl',f"{evalMethod}" ])
        print(model)
        for rep in representations:
            df = pd.read_excel(datapath,sheet_name=rep)
            if mode == "exact3":
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                print(f"Non above {numofclasses} found for {model}")
                continue
            ind = tempdf.iloc[np.argmax(vals)].name
            for col in df_new.keys():
                if col == evalMethod:
                    df_new.loc[rep][col] = df[f"{model}_{evalMethod}"][ind]
                elif col == 'num_of_cl':
                    df_new.loc[rep][col] = df[f"{col}_{model}"][ind]
                else:
                    df_new.loc[rep][col] = df[col][ind]
        df_new.to_excel(writer,sheet_name=f"{model}")
    for model in models:
        if "link_com_" not in model: continue
        df_new = pd.DataFrame(index = representations, columns = ['representation','threshold','similarity','num_of_nodes',
                         'num_of_edges','num_of_cl',f"{evalMethod}" ])
        print(model)
        for rep in representations:
            df = pd.read_excel(datapath,sheet_name=rep)
            if mode == "exact3":
                tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
            elif mode == "more3":
                tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
            vals = tempdf[f"{model}_{evalMethod}"].values
            if len(vals)==0:
                print(f"Non above {numofclasses} found for {model}")
                continue
            ind = tempdf.iloc[np.argmax(vals)].name
            for col in df_new.keys():
                if col == evalMethod:
                    df_new.loc[rep][col] = df[f"{model}_{evalMethod}"][ind]
                elif col == 'num_of_cl':
                    df_new.loc[rep][col] = df[f"{col}_{model}"][ind]
                else:
                    df_new.loc[rep][col] = df[col][ind]
        df_new.to_excel(writer,sheet_name=f"{model}")
    writer.save()

def best_ths_forreps(dataset,dataversion,mode,netmodel,models,representations):
    dict = {}
    for rep in representations:
        dict[rep]=[]
        for model in models:
            if model in ['kmeans','average_linkage','lda','ptm','nfm',
                         # 'Louvain_modularity'
                         ]:
                continue
            try:
                df = pd.read_excel(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_{rep}.xlsx",
                                   sheet_name=f"{model}")
            except:
                continue
            ind = np.argmax(df['threshold'].values)
            dict[rep].append((df['threshold'][ind],model))
            # if "link" in model:
            #     dict[rep].append((df['threshold'][ind],model))
            # else:
            #     dict[rep].append(df['threshold'][ind])
    return dict


if __name__=="__main__":
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
    evalMethod = ["element-centric","fmeasure","randIndex","f1-mean","jacardIndex"][0]
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
    # index = [model for model in models if "link_com" not in model] + ['link_comm']
    datapath = f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}.xlsx"
    # eachrep_eachmodel(dataset,dataversion,datapath,netmodel,mode,models,representations)
    d = best_ths_forreps(dataset,dataversion,mode,netmodel,models,representations)

    #
    # writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_besfofreps-for-models.xlsx",
    #                         engine='xlsxwriter')
    # # for rep in representations:
    # #     df = pd.read_excel(datapath,sheet_name=rep)
    # #     writer = pd.ExcelWriter(f"C:/Users/RAKA/Documents/text clustring/data/output/{dataset}/{dataversion}/{dataset}_{dataversion}_report_{netmodel}_{mode}_{rep}.xlsx",
    # #                         engine='xlsxwriter')
    #
    #     # if representations.index(rep)==0:
    #     #     df = pd.read_excel(datapath,sheet_name=rep)
    #     # else:
    #     #     df = pd.concat([df,pd.read_excel(datapath,sheet_name=rep)], join = 'outer').reset_index(drop=True)
    #
    # # for evalMethod in evalMethods:
    # #     df_new = pd.DataFrame(index = [model for model in models if "link_com" not in model] + ['link_comm'], columns = ['representation','threshold',
    # #                                                      'similarity','num_of_nodes',
    # #                                                      'num_of_edges','num_of_cl',
    # #                                                      f"{evalMethod}" ])
    # for model in models:
    #     if "link_com_" in model: continue
    #     df_new = pd.DataFrame(index = representations, columns = ['representation','threshold','similarity','num_of_nodes',
    #                      'num_of_edges','num_of_cl',f"{evalMethod}" ])
    #     print(model)
    #     for rep in representations:
    #         df = pd.read_excel(datapath,sheet_name=rep)
    #
    #         if mode == "exact3":
    #             tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
    #         elif mode == "more3":
    #             tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
    #
    #         vals = tempdf[f"{model}_{evalMethod}"].values
    #         if len(vals)==0:
    #             print(f"Non above {numofclasses} found for {model}")
    #             # df_new.loc[model] = f"Non above {numofclasses} found"
    #             continue
    #
    #
    #         ind = tempdf.iloc[np.argmax(vals)].name
    #         # for ind in list(tempdf.index):
    #         for col in df_new.keys():
    #             if col == evalMethod:
    #                 # df_new[col][model] = df[f"{model}_{evalMethod}"][ind]
    #                 df_new.loc[rep][col] = df[f"{model}_{evalMethod}"][ind]
    #             elif col == 'num_of_cl':
    #                 df_new.loc[rep][col] = df[f"{col}_{model}"][ind]
    #             else:
    #                 df_new.loc[rep][col] = df[col][ind]
    #     df_new.to_excel(writer,sheet_name=f"{model}")
    # for model in models:
    #     if "link_com_" not in model: continue
    #     df_new = pd.DataFrame(index = representations, columns = ['representation','threshold','similarity','num_of_nodes',
    #                      'num_of_edges','num_of_cl',f"{evalMethod}" ])
    #     print(model)
    #     for rep in representations:
    #         df = pd.read_excel(datapath,sheet_name=rep)
    #
    #         if mode == "exact3":
    #             tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
    #         elif mode == "more3":
    #             tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
    #
    #         vals = tempdf[f"{model}_{evalMethod}"].values
    #         if len(vals)==0:
    #             print(f"Non above {numofclasses} found for {model}")
    #             # df_new.loc[model] = f"Non above {numofclasses} found"
    #             continue
    #
    #
    #         ind = tempdf.iloc[np.argmax(vals)].name
    #         # for ind in list(tempdf.index):
    #         for col in df_new.keys():
    #             if col == evalMethod:
    #                 # df_new[col][model] = df[f"{model}_{evalMethod}"][ind]
    #                 df_new.loc[rep][col] = df[f"{model}_{evalMethod}"][ind]
    #             elif col == 'num_of_cl':
    #                 df_new.loc[rep][col] = df[f"{col}_{model}"][ind]
    #             else:
    #                 df_new.loc[rep][col] = df[col][ind]
    #     df_new.to_excel(writer,sheet_name=f"{model}")
    # writer.save()
    #
    # #     maxindval = 0
    # #     maxdf_new = None
    # #     for model in models:
    # #
    # #         if "link_com_" not in model: continue
    # #         if mode == "exact3":
    # #             tempdf= df[df[f"num_of_cl_{model}"]==numofclasses]
    # #         elif mode == "more3":
    # #             tempdf= df[df[f"num_of_cl_{model}"]>=numofclasses]
    # #
    # #         vals = tempdf[f"{model}_{evalMethod}"].values
    # #         if len(vals)==0:
    # #             print(f"Non above {numofclasses} found for {model}")
    # #             # df_new.loc[model] = f"Non above {numofclasses} found"
    # #             continue
    # #
    # #         ind = tempdf.iloc[np.argmax(vals)].name
    # #         # if df[f"{model}_{evalMethod}"][ind] > maxindval:
    # #         df_new = pd.DataFrame(index = tempdf.index , columns = ['threshold','similarity','num_of_nodes',
    # #                  'num_of_edges','num_of_cl',f"{evalMethod}" ])
    # #         # maxindval = df[f"{model}_{evalMethod}"][ind]
    # #         for ind in list(tempdf.index):
    # #             for col in df_new.keys():
    # #                 if col == evalMethod:
    # #                     df_new.loc[ind][col] = df[f"{model}_{evalMethod}"][ind]
    # #                 elif col == 'num_of_cl':
    # #                     df_new.loc[ind][col] = df[f"{col}_{model}"][ind]
    # #                 else:
    # #                     df_new.loc[ind][col] = df[col][ind]
    # #         df_new.to_excel(writer,sheet_name=f"{model}")
    # #     # df_new.to_excel(writer,sheet_name=f"link_com")
    # #
    # #     writer.save()
    # #     # df_new.to_excel(writer, sheet_name=f"{evalMethod}")
    # # # writer.save()
