import os
import pandas as pd
import numpy as np
import re

if __name__=="__main__":

    # path = f"data/datasets/bbc/bbc_raw/business/001.txt"
    path = f"data/datasets/bbc/bbc_raw"
    savepath = f"data/bbc.csv"
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
                # df.loc[i]['text'] = text
                # df.loc[i]['class'] = classDict[topic]
                # df.loc[i]['class_label'] = topic
    df.to_csv(savepath)

    # with open(datapath,'r') as f :
    #     data = f.read()
