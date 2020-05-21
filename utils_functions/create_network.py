import networkx as nx
from time import time as tm
# from utils_functions.multi_processing import multi_process , one_processing_func
from utils_functions.utils import check_print
from multiprocessing import Process, Manager

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

def knn(data,k):
    t1=tm()
    g = nx.Graph()
    c = 0
    cols = list(data.columns)
    indxs = list(data.index)
    for i in cols:
        c= c +1
        check_print(c)
        kbest_neighs = list(data[i].sort_values()[::-1][1:k+1].index)
        for j in kbest_neighs:
            weight = data[i][j]
            g.add_edge(i,j,weight=weight)
    print(f"network creation took {tm()-t1} s")
    return g


def enn(data,threshold):
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

def func(start,end,c,data,threshold,return_list):
    for i in range(start,end):
        # print(i)
        if (i-start)%10==0:print(f"process {c}: {i} {i-start}/{end-start}")
        for j in range(i+1,end):
            print(j)
            weight = similarity_1by1(data[i-start],data[j-start])
            if weight > threshold:
                return_list.append((i,j,weight))

def processes_bounds(cores,c,prev_end,lendata):
    step = int(lendata/cores)
    start = prev_end
    if c != cores -1:
        end = start + step
    else:
        end = lendata
    return start , end

def create_network_1by1(data,threshold,cores):
    t1=tm()
    g = nx.Graph()
    # cores = 2
    return_list =  Manager().list()
    lendata = len(data)
    end = 0
    processes = []
    for c in range(cores):
        start , end =  processes_bounds(cores,c,end,lendata)
        p = Process(target=func,args=(start,end,c,data[start:end],threshold,return_list))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for e in return_list:
        g.add_edge(e[0],e[1],weight=e[2])
    print(f"network creation took {tm()-t1} s")
    return g

