import tensorflow as tf
import tensorflow_hub as hub
import os
from utils_functions.utils import *
from sentence_transformers import SentenceTransformer


# def model(conf):
#     # Create graph and finalize (finalizing optional but recommended).
#     g = tf.Graph()
#     with g.as_default():
#       # We will be feeding 1D tensors of text into the graph.
#       text_input = tf.placeholder(dtype=tf.string, shape=[None])
#       embed = hub.Module(conf["useModule_path"])
#       embedded_text = embed(text_input)
#       init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#     g.finalize()
#     # Create session and initialize.
#     sess = tf.Session(graph=g)
#     sess.run(init_op)
#     return sess , embed , embedded_text, text_input


# def compute_USE_embeddings(data ,conf):
#     total_data = []
#     print(len(data))
#     j=0
#     if len(data)>conf['bachsize'] : i=conf['bachsize']
#     else: i=len(data)
#     sess , embed , embedded_text, text_input = model(conf)
#     while i <= len(data):
#         print( i , " " ,j)
#         emb_data = sess.run(embedded_text, feed_dict={text_input: data[j:i]})
#         total_data.extend(np.around(emb_data.tolist(),6).tolist())
#         j=i
#         if i +conf['bachsize'] <= len(data): i = i + conf['bachsize']
#         elif i >= len(data): break
#         else: i =len(data)
#     return total_data


def prepare_data(datapath,idspath):
    with open(datapath,'r') as f:
        lns = f.readlines()
    with open(idspath,'r') as f:
        ids = f.readlines()
    lns = [x[:-1] for x in lns]
    ids = [x[:-1] for x in ids]
    return lns , ids

def model(data):
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    emb_data = embedder.encode(data)
    return emb_data

def compute_BERT_embeddings(data,ids,conf):
    emb_data = model(data)
    return emb_data ,ids

if __name__=="__main__":
    # dataset = "bbcsport"
    # dataset = "mixed"
    # dataset = "bbcsport-small"
    # dataset = "cran-cisi-pubmed"
    # dataset = "cran_cisi"
    # dataset = "pubmed_cisi"
    # dataset = "cran_pubmed"
    dataset = "twitter-cor2"
    # dataset = "twitter"
    data_version = "V01"
    conf = {
        # 'data_version':"V_01",
        'bachsize':1000,
        'useModule_path':f"C:\\Users\\RAKA\\Documents\\sentence_embedding\\data\\module\\USE",
        'datapath':f"data/output/{dataset}/{data_version}/{dataset}_corpus_{data_version}.txt",
        'idspath':f"data/output/{dataset}/{data_version}/{dataset}_corpusIds_{data_version}.txt",
        'savepath':f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_bert.pkl"
    }
    if "representation" not in os.listdir("/".join(conf['savepath'].split("/")[:4])):
        os.mkdir(os.path.join("/".join(conf['savepath'].split("/")[:4]),"representation"))

    data , ids = prepare_data(conf["datapath"],conf["idspath"])
    # # data = data[:1001]
    emb_data, ids = compute_BERT_embeddings(data,ids ,conf)
    save_data(conf["savepath"][:-4]+".pkl",[emb_data,ids])

