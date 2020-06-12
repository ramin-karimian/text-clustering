import pickle
import pandas as pd
import re
import numpy as np
from data.scripts.utils import *

def create_topical_rep(dataset,data_version,num_cl=3):
    modelnames = ['PTM','LDA']
    ntopicss = ['100','50','20',f'{num_cl}']
    c = 1
    for modelname in  modelnames:
        for ntopics in ntopicss:
            model_version = str(c)
            datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
            # topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_V{ntopics}_{model_version}.theta"
            topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_{dataset}_{data_version}_T{ntopics}_{model_version}.theta"
            # savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{modelname}{ntopics}-{model_version}.pkl"
            savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{modelname}{ntopics}.pkl"
            df = load_data(datapath)
            with open(topicspath,'r') as f:
                lns = f.readlines()
                assert len(df) == len(lns) , f"data len error, check ids {len(lns)} {len(df)} {model_version} {ntopics} {modelname}"
                ids = []
                data = []
                for x in lns:
                    ids.append(len(ids))
                    l = []
                    for y in x[:-2].split(" "):
                        l.append(float(y))
                    data.append(l)
            save_data(savepath,[data,ids])
            c += 1

if __name__=="__main__":
    # dataset = "twitter"
    # dataset = "twitter-cor2"
    # dataset = "mixed"
    # dataset = "mixedOverlap"
    # dataset = "cran-cisi-pubmed-1000"
    # dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = "cran-cisi-pubmed"
    data_version = "V02"
    modelname = 'LDA'
    # modelname = 'PTM'
    ntopics = f"3"
    model_version = f"8"
    # datapath = f"data/{dataset}_preprocessed_data_{data_version}.pkl"
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    # topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_V{ntopics}_{model_version}.theta"
    topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_{dataset}_{data_version}_T{ntopics}_{model_version}.theta"
    # savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{modelname}{ntopics}-{model_version}.pkl"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{modelname}{ntopics}.pkl"
    df = load_data(datapath)
    with open(topicspath,'r') as f:
        lns = f.readlines()
        assert len(df) == len(lns) , "data len error, check ids"
        ids = []
        data = []
        for x in lns:
            ids.append(len(ids))
            l = []
            for y in x[:-2].split(" "):
                l.append(float(y))
            data.append(l)
    save_data(savepath,[data,ids])
