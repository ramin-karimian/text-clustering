import pickle
from data.scripts.utils import *
import os
import pandas as pd
import re

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

if __name__=="__main__":

    data_version="V01"
    dataset = "bbc"
    datapath = f'../{dataset}_preprocessed_data_{data_version}.pkl'
    # embpath=f"C:/Users/RAKA/Documents/metro_data/data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"
    # if oneOrTotal not in os.listdir(f"output_data"): os.mkdir(f"output_data/{oneOrTotal}")
    if data_version not in os.listdir(f"../output"): os.mkdir(f"../output/{data_version}")
    corpuspath=f"../output/{data_version}/{dataset}_corpus_{data_version}.txt"
    corpusIdspath=f"../output/{data_version}/{dataset}_corpusIds_{data_version}.txt"
    datasavepath=f"../output/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"

    df = load_data(datapath)

    # emb = load_data(embpath)
    w2i, ids,lines = convert_corpus(df,corpuspath,corpusIdspath)
    # new_emb,notInEmb = convert_emb(w2i , emb , wordVectorspath)
    # del emb
    save_data(datasavepath,df)
    df.to_excel(datasavepath[:-4]+'.xlsx')
