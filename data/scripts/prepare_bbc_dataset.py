import os
import pandas as pd
import numpy as np
import re

if __name__=="__main__":
    path = f"../datasets/bbc/bbc_raw"
    savepath = f"../bbc.csv"
    df = pd.DataFrame(columns=['text','class','class_label'])
    classDict = {"business":0,"entertainment":1,"politics":2,"sport":3,"tech":4}
    i = 0
    for topic in os.listdir(path):
        for filename in os.listdir(os.path.join(path,topic)):
            with open(os.path.join(path,topic,filename),'r') as f:
                if i%1000 == 0:
                    print(i)
                text = f.read()
                # text = re.split(r"\n+",text)
                text = re.sub(r".\n+",'.  ',text)
                df.loc[i] = text,classDict[topic],topic
                i = i+1
    df.to_csv(savepath)
