import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from utils_functions.utils import *
from time import time as tm

def modif(g,closeness_centrality):
    closeness_centrality_modified={}
    ns =list(g.nodes())
    i=0
    for k,v in closeness_centrality.items():
        closeness_centrality_modified[ns[i]]=v
        i= i +1
    return closeness_centrality_modified

def my_closeness_centrality(G):
    A = nx.adjacency_matrix(G).tolil()
    D = scipy.sparse.csgraph.floyd_warshall(A, directed=False, unweighted=False)
    n = D.shape[0]
    closeness_centrality = {}
    for r in range(0, n):
    # for r in g.nodes():
        cc = 0.0
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter(lambda x: not x[1] == np.inf, possible_paths))
        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
        closeness_centrality[r] = cc
    closeness_centrality = modif(G,closeness_centrality)
    return closeness_centrality

if __name__=="__main__":
    datapath = "../models/lda_model_30_one_article/network_topic_model_0.3th.pkl"
    _,g = load_data(datapath)
    t1= tm()
    my_closeness_centrality = my_closeness_centrality(g)
    t2=tm()
    print(t2-t1)
    nx_closeness_centrality = nx.closeness_centrality(g)
    t3=tm()
    print(t3-t2)
    print("my_closeness_centrality == nx_closeness_centrality " , my_closeness_centrality ==nx_closeness_centrality)
    # closeness_centrality1={}
    # ns =list(g.nodes())
    # i=0
    # for k,v in my_closeness_centrality.items():
    #     closeness_centrality1[ns[i]]=v
    #     i= i +1
