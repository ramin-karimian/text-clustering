from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import  greedy_modularity_communities , label_propagation_communities
import community
from time import time as tm

def Louvain_modularity_community(g,return_dict):
    t1=tm()
    partition = community.best_partition(g)
    for k,v in partition.items():
        return_dict[k]["Louvain_modularity"] = v
    # com = label_propagation_communities(g)
    print(community.best_partition.__name__,f" took {tm()-t1} s")
    return  return_dict

# def Louvain_modularity_community(g,return_dict):
#     t1=tm()
#     partition = community.best_partition(g)
#     try:
#         for k,v in partition.items():
#             return_dict[k]["Louvain_modularity"] = v
#         # com = label_propagation_communities(g)
#     except:
#         return [partition,k,v,return_dict]
#     print(community.best_partition.__name__,f" took {tm()-t1} s")
#     return  return_dict

def girvan_newman_community(g,return_dict):
    centrality_communities = girvan_newman(g, most_valuable_edge=None)
    return centrality_communities

def greedy_modularity_community(g,return_dict):
    t1=tm()
    com = greedy_modularity_communities(g)
    com = list(com)
    for i in range(len(com)):
        for j in com[i]:
            return_dict[j]["greedy_modularity_communities"] = i
    # com = label_propagation_communities(g)
    print(greedy_modularity_communities.__name__,f" took {tm()-t1} s")

