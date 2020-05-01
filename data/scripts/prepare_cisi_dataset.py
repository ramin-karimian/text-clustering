import pandas as pd
import re
from data.datasets.cran.run import return_list_of_docs


if __name__=="__main__":
    datapath = f"../datasets/cisi-a-dataset-for-information-retrieval/CISI.ALL"
    qr = "../datasets/cran/cran.qry"
    savepath = f"../cisi.csv"

    df = pd.DataFrame(columns=['text','class','class_label'])

    absDocs = return_list_of_docs(qr,datapath)
    for text in absDocs:
        text = re.sub(r"\n",' ',text)
        df.loc[len(df)] = text,1,'cisi'
    df.to_csv(savepath)
