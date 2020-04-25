from multiprocessing import Pool
import time
import itertools
import networkx as nx
from utils_functions.utils import *
from time import time as tm

def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)

def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool)*4
    node_chunks = list(chunks(G.nodes(), int(G.order()/node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G]*num_chunks,
                      [True]*num_chunks,
                      [None]*num_chunks,
                      node_chunks))
    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

if __name__ == "__main__":
    datapath = "../models/lda_model_30_total/network_topic_model_0.8th.pkl"
    _,g = load_data(datapath)
    # t1= tm()
    # my_betweenness_centrality = betweenness_centrality_parallel(g)
    # t2=tm()
    # print(t2-t1)
    # nx_closeness_centrality = nx.closeness_centrality(g)
    # t3=tm()
    # print(t3-t2)

    for G in [g]:
        print("")
        print("Computing betweenness centrality for:")
        print(nx.info(G))
        print("\tParallel version")
        start = tm()
        bt = betweenness_centrality_parallel(G)
        print("\t\tTime: %.4F" % (time.time()-start))
        print("\t\tBetweenness centrality for node 0: %.5f" % (bt[24334265]))
        print("\tNon-Parallel version")
        start = tm()
        nx_bt = nx.betweenness_centrality(G)
        print("\t\tTime: %.4F seconds" % (time.time()-start))
        print("\t\tBetweenness centrality for node 0: %.5f" % (nx_bt[24334265]))
        print("betweenness_centrality_parallel == nx.betweenness_centrality " , bt ==nx_bt)
    print("")



# if __name__ == "__main__":
#     G_ba = nx.barabasi_albert_graph(1000, 3)
#     G_er = nx.gnp_random_graph(1000, 0.01)
#     G_ws = nx.connected_watts_strogatz_graph(1000, 4, 0.1)
#     for G in [G_ba, G_er, G_ws]:
#         print("")
#         print("Computing betweenness centrality for:")
#         print(nx.info(G))
#         print("\tParallel version")
#         start = time.time()
#         bt = betweenness_centrality_parallel(G)
#         print("\t\tTime: %.4F" % (time.time()-start))
#         print("\t\tBetweenness centrality for node 0: %.5f" % (bt[0]))
#         print("\tNon-Parallel version")
#         start = time.time()
#         bt = nx.betweenness_centrality(G)
#         print("\t\tTime: %.4F seconds" % (time.time()-start))
#         print("\t\tBetweenness centrality for node 0: %.5f" % (bt[0]))
#     print("")
