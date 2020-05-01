import pandas as pd
import xml.etree.ElementTree as et
from data.scripts.utils import *


if __name__=="__main__":
    datapath = f"../datasets/pubmed20n0001.xml/pubmed20n0001.xml"
    savepath = f"../pubmed.csv"
    xtree = et.parse(datapath)
    xroot = xtree.getroot()
    keep_in_range = [150,200]

    df_cols = ["text", "title", "id",'class','class_label']
    rows = []

    for node in xroot:
        # s_name = node.attrib.get("name")
        if node.find('MedlineCitation').find('Article').find('Abstract'):
            text = node.find('MedlineCitation').find('Article').find('Abstract').find("AbstractText").text if node is not None else None
        else: continue
        if not keep_in_range[0] < len(text.split()) < keep_in_range[1]:
            continue
        id = node.find('PubmedData').find("ArticleIdList").find("ArticleId").text if node is not None else None
        title = node.find('MedlineCitation').find('Article').find('ArticleTitle').text if node is not None else None
        rows.append({"text": text, "title": title,"id": id , 'class':2,'class_label':'pubmed'})

    df = pd.DataFrame(rows, columns = df_cols)
    df.to_csv(savepath)

