from multiprocessing import Process, Manager
from time import time as tm


def centrality_func(g,f,return_dict):
    t1 =tm()
    # print(f.__name__," started")
    res = f(g)
    # print(f.__name__," finished res")
    # d={}
    # for k,v in list(res.items()):
    #     d[k]=v
    # print(f.__name__ , f" took {tm()-t1} s")
    # return_dict[f.__name__]=d
    print(f.__name__ , f" took {tm()-t1} s")
    return_dict[f.__name__]=res

def centrality_multi_process(funcList,argList=None):
    processes=[]
    manager= Manager()
    return_dict = manager.dict()
    func = centrality_func
    for f in funcList:
        # print(f.__name__)
        p=Process(target=func,args=(argList[0],f,return_dict))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return return_dict

def network_func(arglist,f,return_list):
    # print(arglist)
    threshold ,data = arglist
    # t1 =tm()
    res = f(data,threshold)
    # print(f.__name__ , f" took {tm()-t1} s")
    # return_dict[f.__name__]=d
    print(len(return_list))
    return_list.extend(res)

def network_multi_process(funcList,argList=None):
    argList.append(None)
    processes=[]
    manager= Manager()
    return_list = manager.list()
    func = network_func
    for f in funcList:
        argList[1]=f[1]
        p=Process(target=func,args=(argList,f[0],return_list))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return return_list



def one_processing_func(g,f,return_dict,processes=None):
    t1 =tm()
    res = f(g,processes=processes)
    d={}
    for k,v in list(res.items()):
        d[k]=v
    print(f.__name__ , f" took {tm()-t1} s")
    return_dict[f.__name__]=d
    return return_dict
