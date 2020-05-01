import tensorflow as tf
import tensorflow_hub as hub
import os
from utils_functions.utils import *




def prepare_data(datapath,idspath):
    # with open(datapath,'r') as f:
    #     lns = f.readlines()
    # with open(idspath,'r') as f:
    #     ids = f.readlines()
    df = load_data(datapath)
    # lns = [x[:-1] for x in lns]
    data = [" ".join(x) for x in df['tokens']]
    # data = data[2000:]
    ids = range(len(data))
    return data , None

def model(data,embed):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),
                 tf.tables_initializer()])
        emb_data = sess.run(embed(data))
    return emb_data


def compute_USE_embeddings(data ,embed):
    # data , ids = prepare_data(conf["datapath"],conf["idspath"])
    total_data = []
    print(len(data))
    j=0
    if len(data)>1000 : i=1000
    else: i=len(data)
    # for i in range(1000,len(data),1000):
    while i <= len(data):
        print( i , " " ,j)
        emb_data = model(data[j:i],embed)
        # total_data.extend(emb_data.tolist())
        total_data.extend(np.around(emb_data.tolist(),6).tolist())
        j=i
        if i +1000 <= len(data): i = i + 1000
        elif i >= len(data): break
        else: i =len(data)
    # total_data = np.asarray(total_data)
    return total_data

if __name__=="__main__":
    # dataset = "bbcsport"
    # dataset = "cran_cisi_pubmed"
    # dataset = "cran_cisi"
    # dataset = "pubmed_cisi"
    # dataset = "cran_pubmed"
    dataset = "twitter"
    data_version = "V02"
    conf = {
        # 'data_version':"V_01",
        'useModule_path':f"C:\\Users\\RAKA\\Documents\\sentence_embedding\\data\\module\\USE",
        'datapath':f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl",
        # 'datapath':f"data/bbc_preprocessed_data_V01.pkl",
        'idspath':f"data/bbc_preprocessed_dataids_V01.pkl",
        # 'savepath':f"data/bbc_preprocessed_data_V01_use.pkl"
        'savepath':f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_use.pkl"
    }
    if "representation" not in os.listdir("/".join(conf['savepath'].split("/")[:4])):
        os.mkdir(os.path.join("/".join(conf['savepath'].split("/")[:4]),"representation"))
    embed = hub.Module(conf["useModule_path"])

    # emb_data = model(data,embed)
    data , ids = prepare_data(conf["datapath"],conf["idspath"])
    # data = data[:1001]
    emb_data = compute_USE_embeddings(data ,embed)
    save_data(conf["savepath"][:-4]+".pkl",[emb_data,ids])

