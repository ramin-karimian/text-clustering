import pickle
import pandas as pd
import re
import numpy as np
from data.scripts.utils import *


if __name__=="__main__":
    # dataset = "twitter"
    dataset = "twitter-cor2"
    # dataset = "mixed"
    # dataset = "cran-cisi-pubmed"
    data_version = "V01"
    modelname = 'LDA'
    ntopics = f"20"
    model_version = f"21"
    # datapath = f"data/{dataset}_preprocessed_data_{data_version}.pkl"
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    # topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_V{ntopics}_{model_version}.theta"
    topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_{dataset}_{data_version}_T{ntopics}_{model_version}.theta"
    # savepath = f"data/{dataset}_preprocessed_data_{data_version}_{modelname}{model_version}.pkl"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{modelname}{ntopics}-{model_version}.pkl"
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
