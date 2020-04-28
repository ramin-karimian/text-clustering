import os
import pandas as pd
import numpy as np
import re

if __name__=="__main__":
    path = f"../datasets/bbcsport/bbcsport_raw"
    savepath = f"../bbcsport.csv"
    df = pd.DataFrame(columns=['text','class','class_label'])
    classDict = {"athletics":0,"cricket":1,"football":2,"rugby":3,"tennis":4}
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
