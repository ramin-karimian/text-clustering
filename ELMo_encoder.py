import tensorflow as tf
import tensorflow_hub as hub
import os
import h5py
from utils_functions.utils import *



def prepare_data(datapath,idspath):
    with open(datapath,'r') as f:
        lns = f.readlines()
    with open(idspath,'r') as f:
        ids = f.readlines()
    lns = [x[:-1] for x in lns]
    ids = [x[:-1] for x in ids]
    return lns , ids

# def model(data,embed):
#     with tf.Session() as sess:
#         sess.run([tf.global_variables_initializer(),
#                  tf.tables_initializer()])
#         emb_data = sess.run(embed(data,as_dict=True)['elmo'])
#     return emb_data

# def compute_ELMO_embeddings(data,ids,embed,conf):
#     emb_data = model(data,embed)
#     emb_data = [x.tolist() for x in emb_data]
#     return emb_data ,ids


def elmo_vectors_elmo(embeddings):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tf.reduce_mean(embeddings,1))

def elmo_vectors_default(embeddings):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(embeddings)


# def model(conf):
#     # Create graph and finalize (finalizing optional but recommended).
#     g = tf.Graph()
#     with g.as_default():
#       # We will be feeding 1D tensors of text into the graph.
#       text_input = tf.placeholder(dtype=tf.string, shape=[None])
#       embed = hub.Module(conf["module_path"])
#       embedded_text = embed(text_input)
#       init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#     g.finalize()
#     # Create session and initialize.
#     sess = tf.Session(graph=g)
#     sess.run(init_op)
#     return sess , embed , embedded_text, text_input
#
# def compute_BERT_embeddings(data ,conf):
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
        'bachsize':15,
        'module_path':f"data/scripts/modules/tfhub.devgoogleelmo2",
        'datapath':f"data/output/{dataset}/{data_version}/{dataset}_corpus_{data_version}.txt",
        'idspath':f"data/output/{dataset}/{data_version}/{dataset}_corpusIds_{data_version}.txt",
        # 'savepath':f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_elmo.pkl"
        'savepath':f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}"
    }
    if "representation" not in os.listdir("/".join(conf['savepath'].split("/")[:4])):
        os.mkdir(os.path.join("/".join(conf['savepath'].split("/")[:4]),"representation"))

    # data , ids = prepare_data(conf["datapath"],conf["idspath"])
    # total_data = compute_BERT_embeddings(data ,conf)

    # embed = hub.Module(conf["module_path"],trainable=False)
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    data , ids = prepare_data(conf["datapath"],conf["idspath"])
    # data = data[:2]
    steps = int(len(data)/conf['bachsize'])
    prev_end = 0
    elmo_emb = []
    def_emb = []
    for s in range(steps):
        start = prev_end
        if s != steps-1:
            end = start + conf['bachsize']
        else:
            end = len(data)
        prev_end = end
        print(f" s {s} _ satrt {start} _ end {end}")

        embeddings = elmo(data[start:end], signature="default", as_dict=True)
        elmo_emb.extend([x.tolist() for x in elmo_vectors_elmo(embeddings["elmo"])])
        def_emb.extend([x.tolist() for x in elmo_vectors_default(embeddings["default"])])

    # # # data = data[:1001]
    # emb_data, ids = compute_ELMO_embeddings(data,ids,embed ,conf)
    save_data(conf["savepath"]+"_elmo-weighted.pkl",[elmo_emb,ids])
    save_data(conf["savepath"]+"_elmo-default.pkl",[def_emb,ids])

