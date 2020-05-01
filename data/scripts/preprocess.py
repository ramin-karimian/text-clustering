import time
from data.scripts.utils import *
from data.scripts.preprocess_utils import *

if __name__ == '__main__':
    # dataset = "pubmed"
    dataset = "twitter"
    confiq = {
        "datapath" : f"../{dataset}.csv",
        # "datapath" : "../cisi.csv",
        # "datapath" : "source_data/Tags.xlsx",
        # "datapath" : "source_data/28_article_itself.txt",
        "preprocessed_datapath" : f"../{dataset}_preprocessed_data",
        "data_version":"V02",
        # 'remove_pos_tags':[],
        'remove_pos_tags':['DT','CC','IN','PRP','PRP$','MD','WDT','WP','WP$','WRB'],
        # "data_version":"28Article_V02",
        # 'artId':'nyt://article/f7ca9bef-99d8-58cd-a394-1ab584c3dd25',
        "cores" : 14,
        # "usecols" : ["commentBody","commentID","parentID","articleID"]
        # "usecols" : ["text","class",'class_label']
        # "usecols" : ["text","class",'class_label']
        "usecols" : ["text","class",'class_label',"tweet_id"]
    }

    starttime = time.time()
    data=pd.read_csv(confiq["datapath"],usecols=confiq["usecols"])
    # data=pd.read_excel(confiq["datapath"],usecols=confiq["usecols"])

    # with open(confiq["datapath"],encoding="utf8") as f:
    #     ln = f.read()
    #     data = pd.DataFrame([[ln,confiq["artId"]]],columns=['Comment','commentID'],index=[0])

    # data=data.iloc[:100]
    lenData=len(data)

    data = preprocess_multi_process(preprcess_func,confiq["cores"],data,confiq['remove_pos_tags'])

    data.to_csv(f'{confiq["preprocessed_datapath"]}_{confiq["data_version"]}.csv')
    save_data(f'{confiq["preprocessed_datapath"]}_{confiq["data_version"]}.pkl',data)

    print('That took {} seconds'.format(time.time() - starttime))



