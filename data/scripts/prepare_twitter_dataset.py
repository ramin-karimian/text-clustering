import re
import pandas as pd
import numpy as np
import os
from utils_functions.utils  import *

if __name__=="__main__":
    path = '../datasets/twitter'
    df = pd.DataFrame(columns=['text','class','class_label'])
    label = 0
    c=0
    for f in os.listdir(path):
        dfch = pd.read_csv(os.path.join(path,f),header = None)
        dfch.columns = ['date','tweet_id','text' ]

        dfch['text'] = dfch['text'].apply(lambda x:x[2:-1])
        dfch['class'] = f.split('.')[0][1:]
        dfch['class_label'] = label
        label = label + 1
        df = pd.concat([df,dfch])
        c= c + len(dfch)
    df.to_csv('../twitter.csv')
