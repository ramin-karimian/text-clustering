import pickle
import pandas as pd
import re
import numpy as np
from data.scripts.utils import *


if __name__=="__main__":
    data_version = "V01"
    modelname = 'wntm'
    ntopics = f"50"
    model_version = f"14"
    # modelname='wntm'
    datapath = f"data/bbc_preprocessed_data_{data_version}.pkl"
    topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_V{ntopics}_{model_version}.theta"
    savepath = f"data/bbc_preprocessed_data_{data_version}_{modelname}{model_version}.pkl"
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
