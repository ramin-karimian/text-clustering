import re
import pandas as pd
import numpy as np
import os
from utils_functions.utils  import *
from sklearn.utils  import shuffle

if __name__=="__main__":
    path = '../datasets/twitter'
    samplenum = -1
    df = pd.DataFrame(columns=['text','class','class_label'])
    label = 0
    c=0
    for f in os.listdir(path):
        dfch = pd.read_csv(os.path.join(path,f),header = None)
        # dfch.columns = ['date','tweet_id','text' ]
        dfch.columns =['date','tweet_id','text' ,"hashtags"]

        dfch['text'] = dfch['text'].apply(lambda x:x[2:-1])
        # dfch['class'] = f.split('.')[0][1:]
        # dfch['class_label'] = label
        dfch['class'] = label
        # dfch['class_label'] = f.split('.')[0][1:]
        dfch['class_label'] = f.split('.')[0]
        label = label + 1
        # df = pd.concat([df,dfch])
        dfch = shuffle(dfch).reset_index(drop=True).iloc[:samplenum]
        df = pd.concat([df,dfch])
        c= c + len(dfch)
    df.to_csv('../twitter.csv',index= None)
    print(df['class_label'].value_counts())
