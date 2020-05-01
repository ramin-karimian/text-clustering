import networkx as nx
from time import time as tm
# from utils_functions.multi_processing import multi_process , one_processing_func
from utils_functions.utils import check_print
from utils_functions.compute_similarity import similarity_1by1


# def create_network_(data,threshold=0):
#     edgeList=[]
#     c = 0
#     # for i in range(len(data)):
#     cols = list(data.columns)
#     indxs = list(data.index)
#     for i in cols:
#
#         # check_print(c)
#         for j in indxs[cols.index(i)+1:]:
#             th = data[i][j]
#             if th > threshold:
#                 c= c +1
#                 edgeList.append((i,j))
#                 # g.add_edge(i,j,th=th)
#     # print(f"network creation took {tm()-t1} s")
#     # return g
#     print("c : ",c)
#     return edgeList

def create_network(data,threshold):
    t1=tm()
    g = nx.Graph()
    c = 0
    # for i in range(len(data)):
    cols = list(data.columns)
    indxs = list(data.index)
    for i in cols:
        c= c +1
        check_print(c)
        for j in indxs[cols.index(i)+1:]:
            weight = data[i][j]
            if weight > threshold:
                g.add_edge(i,j,weight=weight)
    print(f"network creation took {tm()-t1} s")
    return g

def create_network_1by1(data,threshold):
    t1=tm()
    g = nx.Graph()
    c = 0
    for i in range(len(data)):
        c= c +1
        check_print(c)
        # for j in indxs[cols.index(i)+1:]:
        for j in range(i+1,len(data)):
            weight = similarity_1by1(data[i],data[j])
            if weight > threshold:
                g.add_edge(i,j,weight=weight)
    print(f"network creation took {tm()-t1} s")
    return g
