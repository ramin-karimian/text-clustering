import time
from data.scripts.utils import *
from data.scripts.preprocess_utils import *

if __name__ == '__main__':
    # dataset = "pubmed"
    # dataset = "twitter-cor2"
    # dataset = "twitter"
    dataset = "bbcsport-small"
    confiq = {
        "datapath" : f"../{dataset}.csv",
        "preprocessed_datapath" : f"../{dataset}_preprocessed_data",
        "data_version":"V01",
        'remove_pos_tags':['DT','CC','IN','PRP','PRP$','MD','WDT','WP','WP$','WRB'],
        "cores" : 7,
        # "extend_stoplist" : ["art","business","education","food","technology"],
        "extend_stoplist" : [],
        # "usecols" : ["commentBody","commentID","parentID","articleID"]
        "usecols" : ["text","class",'class_label']
        # "usecols" : ["text","class",'class_label',"tweet_id"],
        # "usecols" : ["text","class",'class_label','multi_class',"tweet_id","hashtags"]
    }

    starttime = time.time()
    data=pd.read_csv(confiq["datapath"],usecols=confiq["usecols"])

    # data=data.iloc[:100]
    lenData=len(data)

    data = preprocess_multi_process(preprcess_func,confiq,data)

    data.to_csv(f'{confiq["preprocessed_datapath"]}_{confiq["data_version"]}.csv')
    save_data(f'{confiq["preprocessed_datapath"]}_{confiq["data_version"]}.pkl',data)

    print('That took {} seconds'.format(time.time() - starttime))



