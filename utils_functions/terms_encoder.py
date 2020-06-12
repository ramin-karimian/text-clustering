import pickle
import pandas as pd
import re
import numpy as np
import os
from data.scripts.utils import *

def create_termstf_rep(dataset,data_version):
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_terms-tf.pkl"
    # savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_terms-tfidf.pkl"
    if "representation" not in os.listdir("/".join(savepath.split("/")[:4])):
        os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"representation"))

    data = list(load_data(datapath)['tokens'].values)

    save_data(savepath,data)

if __name__=="__main__":
    # dataset = "bbcsport"
    # dataset = "cran-cisi-pubmed"
    # dataset = "cran-cisi-pubmed-1000"
    # dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = "mixed"
    # dataset = "cran_cisi"
    # dataset = "pubmed_cisi"
    # dataset = "cran_pubmed"
    # dataset = "twitter"
    # dataset = "mixedOverlap"
    # dataset = "twitter-cor2"
    # data_version="V01"
    data_version="V02"
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_terms-tf.pkl"
    # savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_terms-tfidf.pkl"
    if "representation" not in os.listdir("/".join(savepath.split("/")[:4])):
        os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"representation"))

    data = list(load_data(datapath)['tokens'].values)

    save_data(savepath,data)



