from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import pandas as pd
import re
import numpy as np
from data.scripts.utils import *


if __name__=="__main__":
    data_version="V01"
    datapath = f"data/bbc_preprocessed_data_{data_version}.pkl"
    savepath = f"data/bbc_preprocessed_data_{data_version}"
    df = load_data(datapath)
    data = [" ".join(x) for x in df['tokens']]



    vectorizer = TfidfVectorizer()
    tfidf_data = vectorizer.fit_transform(data)
    save_data(savepath+"_tfidf.pkl",tfidf_data)


    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    tf_data = vectorizer.transform(data)
    save_data(savepath+"_tf.pkl",tf_data)



