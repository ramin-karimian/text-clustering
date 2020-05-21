import re
import pandas as pd
import numpy as np
import os
from utils_functions.utils  import *
from sklearn.utils  import shuffle

if __name__=="__main__":
    dataset = "twitter-cor2"
    path = f'../datasets/{dataset}'
    # samplenum = 10000
    samplenum = -1
    # df = pd.DataFrame(columns=['text','class','class_label'])
    df = pd.DataFrame(columns=['text','class','class_label','multi_class'])
    # classDict = {"art":0,"business":1,"education":2,"food":3,"technology":4}
    classDict = {"food":0,"business":1,"technology":2}
    label = 0
    c=0
    for f in os.listdir(path):
        dfch = pd.read_csv(os.path.join(path,f),header = None)
        # dfch.columns = ['date','tweet_id','text' ]
        dfch.columns = ['date','tweet_id','text' ,"hashtags"]
        dfch = dfch.iloc[dfch['text'].dropna().index].reset_index()
        dfch['text'] = dfch['text'].apply(lambda x:x[2:-1])
        # dfch['class'] = f.split('.')[0][1:]
        # dfch['class_label'] = label

        # f = re.sub("#","",f)
        if "-" in f.split('.')[0]:
            labels = " , ".join(f.split('.')[0].split("-"))
            dfch['multi_class'] = " , ".join([str(classDict[x]) for x in labels.split(" , ")])
        else:
            labels = f.split('.')[0]
            dfch['multi_class'] = str(classDict[labels])

        dfch['class_label'] = labels
        dfch['class'] = label
        label = label + 1

        dfch = shuffle(dfch).reset_index(drop=True).iloc[:samplenum]
        df = pd.concat([df,dfch])
        c= c + len(dfch)
    df.to_csv(f'../{dataset}.csv',index= None)
    print(df['class_label'].value_counts())
