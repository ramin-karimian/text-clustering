from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import pandas as pd
import re
import numpy as np
import os
from data.scripts.utils import *

def create_tftfidf_rep(dataset,data_version):
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    # savepath = f"data/bbc_preprocessed_data_{data_version}"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}"
    if "representation" not in os.listdir("/".join(savepath.split("/")[:4])):
        os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"representation"))

    df = load_data(datapath)
    data = [" ".join(x) for x in df['tokens']]

    vectorizer = TfidfVectorizer()
    tfidf_data = vectorizer.fit_transform(data)
    save_data(savepath+"_tfidf.pkl",tfidf_data)

    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    tf_data = vectorizer.transform(data)
    save_data(savepath+"_tf.pkl",tf_data)

if __name__=="__main__":
    # dataset = "bbcsport"
    # dataset = "mixed"
    # dataset = "mixedOverlap"
    # dataset = "cran-cisi-pubmed"
    # dataset = "cran-cisi-pubmed-1000"
    # dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = "cran_cisi"
    # dataset = "pubmed_cisi"
    # dataset = "cran_pubmed"
    # dataset = "twitter"
    # dataset = "twitter-cor2"
    data_version="V02"
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    # savepath = f"data/bbc_preprocessed_data_{data_version}"
    savepath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}"
    if "representation" not in os.listdir("/".join(savepath.split("/")[:4])):
        os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"representation"))

    df = load_data(datapath)
    data = [" ".join(x) for x in df['tokens']]



    vectorizer = TfidfVectorizer()
    tfidf_data = vectorizer.fit_transform(data)
    save_data(savepath+"_tfidf.pkl",tfidf_data)


    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    tf_data = vectorizer.transform(data)
    save_data(savepath+"_tf.pkl",tf_data)



