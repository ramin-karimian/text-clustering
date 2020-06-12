import os
from collections import defaultdict
from multiprocessing import Process,Manager
import subprocess
from time import time as tm
from utils_functions.comm2nodes import write_comm2nodes


def swap(a,b):
    if int(a) > int(b):
        return b,a
    return a,b

def swap_edgepair(e1,e2):
    if int(e1[0])> int(e2[0]):
        return e2,e1
    elif int(e1[0]) == int(e2[0]):
        if int(e1[1])> int(e2[1]):
            return e2,e1
        else:
            return e1,e2
    else:
        return e1,e2


def read_edgelist_unweighted(filename,delimiter=None,nodetype=str):
    adj = defaultdict(set) # node to set of neighbors
    edges = set()
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
        if ni != nj: # skip any self-loops...
            edges.add( swap(ni,nj) )
            adj[ni].add(nj)
            adj[nj].add(ni) # since undirected
    return dict(adj), edges

def read_edgelist_weighted(filename,delimiter=None,nodetype=str,weighttype=float):
    adj = defaultdict(set)
    edges = set()
    ij2wij = {}
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        ni,nj,wij = nodetype(L[0]),nodetype(L[1]),weighttype(L[2]) # other columns ignored
        if ni != nj: # skip any self-loops...
            ni,nj = swap(ni,nj)
            edges.add( (ni,nj) )
            ij2wij[ni,nj] = wij
            adj[ni].add(nj)
            adj[nj].add(ni) # since undirected
    return dict(adj), edges, ij2wij

def convert_pairs_file(conf,filename):
    folder = conf['resultspath']
    filepath = conf['pairsfilepath']
    # filename = filepath.split("/")[-1]
    with open(filepath,'r') as f:
        with open(f"{folder}/modified_unweighted_{filename}",'w') as f1:
            with open(f"{folder}/modified_weighted_{filename}",'w') as f2:
                lns = f.readlines()
                for line in lns:
                    # if conf["is_weighted"]:
                    x = line[:-1].split(",")[:3]
                    f2.write(f"{x[0]} {x[1]} {int(float(x[2])*100)}\n")
                    # else:
                    x = line[:-1].split(",")[:2]
                    f1.write(f"{x[0]} {x[1]}\n")

def comb(x):
    return (x*(x-1))/2

def calc_bounds(cores,p,prev_end,adj,ln_adj):
    start = prev_end
    if p != cores-1 :
        s=0
        end = start
        for x in list(adj.keys())[start:]:
            if comb(len(adj[x])) + s <= ln_adj:
                s = s + comb(len(adj[x]))
                end = end + 1
    elif p == cores-1 : end = len(adj)
    return start,end

def calc_jaccards_cpp(conf,start,end,modified_pairsfilepath):
    jaccards_filepath = f"{conf['resultspath']}/{start}_{end}.txt"
    cmd = [conf['calcJaccards_program'],modified_pairsfilepath,jaccards_filepath]
    prc = subprocess.Popen(cmd)
    res = prc.communicate()

def compute_jaccards_cpp_integrated(conf,filename):
    print ("# loading network from edgelist...")

    if conf["is_weighted"]:
        adj,edges,ij2wij = read_edgelist_weighted(conf['pairsfilepath'], delimiter=conf['delimiter'])
        modified_pairsfilepath = f"{conf['resultspath']}/modified_weighted_{filename}"
    else:
        adj,edges        = read_edgelist_unweighted(conf['pairsfilepath'], delimiter=conf['delimiter'])
        modified_pairsfilepath = f"{conf['resultspath']}/modified_unweighted_{filename}"

    processes = []
    prev_end = 0
    ln_adj = sum([comb(len(adj[x])) for x in adj])
    ln_adj = int(ln_adj/conf['cores'])
    for c in range(conf['cores']):
        # print(f"core num {c} initiated")
        start,end = calc_bounds(conf['cores'],c,prev_end,adj,ln_adj)
        prev_end = end
        p = Process(target=calc_jaccards_cpp,args=(conf,start,end,modified_pairsfilepath))
        p.start()
        # print(f"core num {c} started")
        processes.append(p)
    for p in processes:
        p.join()


def process_jaccards_files_cpp(folder,fname):
    savepath = f"{folder}/{fname[:-6]}.jaccs"
    # with open(savepath,'w') as f1:
    with open(savepath,'a') as f1:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                # print(f"filename: {filename}")
                datapath = f"{folder}/{filename}"
                with open(datapath,'r') as f:
                    # lns = f.readlines()
                    # for line in lns:
                    line = f.readline()
                    doc = ""
                    while line:
                    # for line in  f.readline():
                        item = line[:-1].split("\t")
                        f1.write(f"{int(item[0])}\t{int(item[1])}\t{int(item[2])}\t{int(item[3])}\t{float(item[4])}\n")
                        line = f.readline()
                os.remove(datapath)
    return savepath

def multiprocess_jaccards_files_cpp_func(folder,filename,returndict):
    datapath = f"{folder}/{filename}"
    with open(datapath,'r') as f:
        # lns = f.readlines()
        # for line in lns:
        line = f.readline()
        doc = ""
        while line:
            item = line[:-1].split("\t")
            doc = doc + f"{int(item[0])}\t{int(item[1])}\t{int(item[2])}\t{int(item[3])}\t{float(item[4])}\n"
            line = f.readline()
    returndict[filename] = doc
    os.remove(datapath)


def multiprocess_jaccards_files_cpp(folder,fname):
    returndict = Manager().dict()
    processes = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            # tt = tm()
            process = Process(target=multiprocess_jaccards_files_cpp_func,args=(folder,filename,returndict))
            process.start()
            processes.append(process)
            # t5 = print_time(tt)
    for p in processes:
        p.join()

    keys = returndict.keys()
    keys = sorted(keys, key=lambda x: int(x.split("_")[0]))
    savepath = f"{folder}/{fname[:-6]}.jaccs"

    with open(savepath,'w') as f1:
        for filename in keys:
            doc = returndict[filename]
            f1.write(doc)
    return savepath

def run_jaccs_clustering(conf,th,filename):
    print(f"th: {th}")
    jaccards_filepath = f"{conf['resultspath']}/{filename[:-6]}.jaccs"

    # filename = conf['pairsfilepath'].split("/")[-1]
    modified_pairsfilepath = f"{conf['resultspath']}/modified_unweighted_{filename}" ##cpp in this phase only needs unweighted edges list
    cmd = [conf['clusterJaccards_program'],modified_pairsfilepath,jaccards_filepath,f"{conf['resultspath']}/{filename[:-6]}_{th}.clusters",f"{conf['resultspath']}/{filename[:-6]}_{th}.mc_nc",str(th)]
    prc = subprocess.Popen(cmd)
    res = prc.communicate()
    return res


def jaccs_clustering_multiprocess_func(conf,th,filename,returndict):
    res = run_jaccs_clustering(conf,th,filename)
    cid2nodes = write_comm2nodes(conf,filename,th)
    returndict[th] = cid2nodes


def jaccs_clustering_multiprocess(conf,filename,return_dict):
    returndict = Manager().dict()
    processes = []

    for th in conf["th_list"]:
        # tt = tm()
        process = Process(target=jaccs_clustering_multiprocess_func,args=(conf,th,filename,returndict))
        process.start()
        processes.append(process)
        # t5 = print_time(tt)
    for p in processes:
        p.join()

    keys = returndict.keys()
    keys.sort()
    for th in keys:
        for c,ns in returndict[th].items():
            for n in ns:
                n = int(n)
                if f"link_com_{round(th,2)}cut" in return_dict[n]:
                    return_dict[n][f"link_com_{round(th,2)}cut"] += f",{str(c)}"
                else:
                    return_dict[n][f"link_com_{round(th,2)}cut"] = str(c)
    if conf["delete_jacfile"]:
        os.remove(f"{conf['resultspath']}/{filename[:-6]}.jaccs")
    return return_dict

def print_time(t):
    tt = tm()
    print(f"{(tt-t)/60} mins\n")
    return tt
