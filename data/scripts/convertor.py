from data.scripts.utils import *
import os
import pandas as pd
import numpy as np

def convert_corpus(df,corpuspath,corpusIdspath):
    w2i={}
    lenDf = len(df)
    ids,lines =[],[]
    # lenDf=10
    with open(corpuspath,'w') as f:
        with open(corpusIdspath,'w') as f1:
            for i in range(lenDf):
                line = []
                for j in range(len(df['tokens'][i])):
                    # if df['tokens'][i][j] in emb:
                    line.append(df['tokens'][i][j])
                    if df['tokens'][i][j] not in w2i:
                        w2i[df['tokens'][i][j]]=len(w2i) + 1
                if len(line)==0:
                    print(i )
                    continue
                line = str(" ".join(line)+"\n")
                id = str(i) + "\n"
                # if i != lenDf-1:
                #     line = str(" ".join(line)+"\n")
                #     id = str(df["commentID"][i]) + "\n"
                # else:
                #     line = str(" ".join(line))
                #     id = str(df["commentID"][i])
                f1.write(id)
                ids.append(id)
                f.write(line)
                lines.append(line)
    return w2i, ids,lines

def convert_emb(w2i , emb , wordVectorspath):
    new_emb = {}
    notInEmb = []
    emb_items = emb.items()
    last_k = list(emb.items())[-1][0]
    with open(wordVectorspath,"w") as f:
        w2i_keys = w2i.keys()
        for k,v in emb_items:
            if k in w2i_keys:
                new_emb[k]=v
                f.write(str(k)+" ")
                len_v =len(v)
                for i in range(len_v) :
                    if i != len_v-1:
                        f.write(str(v[i])+" ")
                    else:
                        if k != last_k:
                            f.write(str(v[i])+"\n")
                        else:
                            f.write(str(v[i]))
            else:
                notInEmb.append(k)

    return new_emb,notInEmb

def convert(data_version,dataset,df,rep):
    if dataset not in os.listdir(f"../output"): os.mkdir(f"../output/{dataset}")
    if data_version not in os.listdir(f"../output/{dataset}/"): os.mkdir(f"../output/{dataset}/{data_version}")
    corpuspath=f"../output/{dataset}/{data_version}/{dataset}_corpus_{data_version}.txt"
    corpusIdspath=f"../output/{dataset}/{data_version}/{dataset}_corpusIds_{data_version}.txt"
    datasavepath=f"../output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    w2i, ids,lines = convert_corpus(df,corpuspath,corpusIdspath)
    save_data(datasavepath,df)
    df.to_excel(datasavepath[:-4]+'.xlsx')
    f = open(f"../output/{dataset}/{data_version}/rep.txt",'w')
    f.write('dataset' + " " + 'average' + " " + 'count'+ " " + 'aveStd'+ " "+ 'max'+ " "+ 'min'+"\n")
    for x in rep:
        f.write(x + " " + str(round(rep[x]['aveLen'],0)) + " " + str(rep[x]['count'])+ " " + str(round(rep[x]['aveStd'],0))+ " " + str(rep[x]['max'])+ " " + str(rep[x]['min']) + "\n")
    f.close()
    return df

def combine(dataset,data_version):
    datasets = dataset.split("_")
    df_total = ''
    for d in datasets:
        print(d)
        datapath = f'../{d}_preprocessed_data_{data_version}.pkl'
        df = pd.DataFrame(load_data(datapath),columns= ["text","class",'class_label','tokens'])
        if len(df_total)>0:
            df_total = pd.concat([df_total,df],ignore_index= True)
        else: df_total = df
    return df_total

def remove_low_dimentional_items(df,limit):
    for i in range(len(df['tokens'])):
        if len(df['tokens'][i]) < limit :
            df.drop(i,inplace=True)
    df.index = range(len(df))
    return df

def create_rep(dataset,df):
    rep={}
    rep[dataset] = {}
    rep[dataset]['aveLen']= np.mean([len(x) for x in df['tokens']])
    rep[dataset]['aveStd']= np.std([len(x) for x in df['tokens']])
    rep[dataset]['count'] = len(df)
    rep[dataset]['max']= np.max([len(x) for x in df['tokens']])
    rep[dataset]['min']= np.min([len(x) for x in df['tokens']])
    rep[dataset]['lens']= [len(x) for x in df['tokens']]
    rep[dataset]['count']= len(df)
    return rep

if __name__=="__main__":

    data_version="V03"
    # dataset = "cran_cisi_pubmed"
    # dataset = "bbcsport"
    # dataset = "cran_pubmed"
    # dataset = "pubmed_cisi"
    dataset = "twitter"
    limit = 5
    # limit = 5
    if len(dataset.split("_"))>1:
        df = combine(dataset,data_version)
        df = remove_low_dimentional_items(df,limit)
        rep = create_rep(dataset,df)

    else:
        datapath = f'../{dataset}_preprocessed_data_{data_version}.pkl'
        df = load_data(datapath)
    #     df = remove_low_dimentional_items(df,limit)
    #     rep = create_rep(dataset,df)
    #
    # df = convert(data_version,dataset,df,rep)



