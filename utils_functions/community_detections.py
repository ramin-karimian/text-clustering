from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import  greedy_modularity_communities ,\
    label_propagation_communities, k_clique_communities, asyn_fluidc
import community
from time import time as tm
import infomap
import networkx as nx


def Louvain_modularity_community(g,return_dict,num_cl):
    t1=tm()
    partition = community.best_partition(g)
    for k,v in partition.items():
        return_dict[k]["Louvain_modularity"] = v
    # com = label_propagation_communities(g)
    print(community.best_partition.__name__,f" took {tm()-t1} s")
    return  return_dict


def girvan_newman_community(g,return_dict,num_cl):
    centrality_communities = girvan_newman(g, most_valuable_edge=None)
    return centrality_communities

def greedy_modularity_community(g,return_dict,num_cl):
    t1=tm()
    com = greedy_modularity_communities(g)
    com = list(com)
    for i in range(len(com)):
        for j in com[i]:
            return_dict[j]["greedy_modularity_communities"] = i
    # com = label_propagation_communities(g)
    print(greedy_modularity_communities.__name__,f" took {tm()-t1} s")

def label_propagation_community(g,return_dict,num_cl):
    t1=tm()
    comm = label_propagation_communities(g)
    c = 0
    for v in comm:
        for k in v:
            return_dict[k]["label_propagation"] = c
        c = c + 1
    print(label_propagation_communities.__name__,f" took {tm()-t1} s")
    return  return_dict

def k_clique_community(g,return_dict,num_cl):
    t1=tm()
    comm = k_clique_communities(g,4)
    c = 0
    for v in comm:
        for k in v:
            return_dict[k]["k_clique"] = c
        c = c + 1
    print(k_clique_communities.__name__,f" took {tm()-t1} s")
    return  return_dict

def asyn_fluidc_community(g,return_dict,num_cl):
    t1=tm()
    comm = asyn_fluidc(g, num_cl, max_iter=100, seed=None)
    c = 0
    for v in comm:
        for k in v:
            return_dict[k]["asyn_fluidc"] = c
        c = c + 1
    print(asyn_fluidc.__name__,f" took {tm()-t1} s")
    return  return_dict

def infomap_community(g,return_dict,num_cl):
    t1=tm()
    _ ,communities = find_infomap_communities(g)
    for k,v in communities.items():
        return_dict[k]["infomap"] = v
    print(find_infomap_communities.__name__,f" took {tm()-t1} s")
    return  return_dict


def find_infomap_communities(g):
    im = infomap.Infomap("--two-level --silent")

    for source, target in g.edges:
        im.addLink(source, target)

    im.run()
    # print(f"Found {im.numTopModules()} modules with codelength: {im.codelength()}")
    communities = im.getModules()
    nx.set_node_attributes(g, communities, 'community')
    return g,communities

