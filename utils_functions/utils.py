import pandas as pd
import pickle
import numpy as np
from time import time as tm

def load_data(datapath):
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    return data


def save_data(datapath, data):
    with open(datapath, "wb") as f:
        pickle.dump(data, f)



def load(datapath,extention=False,article="total"):
    with open(datapath, "rb") as f:
        df = pickle.load(f)
    if article == "total":
        if extention == "tokens":
            data = list(df["tokens"])
            return [(data, df)]
        elif extention == "topics":
            data = list(df["topic_distribution"])
            # data = [[y[1] for y in x] for x in data]
            # data = [[float(y) for y in x] for x in data]
            return [(data, df)]
        else:
            return [df]
    elif article=="one_article":
        ID = df["articleID"][0]
        df = df[df["articleID"]==ID]
        if extention == "tokens":
            data = df["tokens"]
            return [(data, df)]
        elif extention == "topics":
            data = list(df["topic_distribution"])
            data = [[y[1] for y in x] for x in data]
            return [(data, df)]
        else:
            return [df]
    elif article=="total_one_article":
        ids = list(df['articleID'].unique())
        dfs = []
        for i in ids:
            # d = df[df["articleID"]==i]
            d = pd.DataFrame(df.loc[df["articleID"]==i])
            if extention == "tokens":
                data = d["tokens"]
                dfs.append((data, d))
                # return data, df
            elif extention == "topics":
                data = list(d["topic_distribution"])
                data = [[y[1] for y in x] for x in data]
                # return data, df
                dfs.append((data, d))
            else:
                dfs.append(d)
                # return df
        return dfs

def check_print(i,step=1000,time=False):
    if i%step==0:
        if time:
            print(i," time(min): ",(tm()-time)/60)
        else:
            print(i)
