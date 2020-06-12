import networkx as nx
from time import time as tm
import os
from utils_functions.utils import *
from utils_functions.create_network import enn,knn
from utils_functions.modify import modify
from utils_functions.multi_processing import centrality_multi_process , network_multi_process , one_processing_func
from utils_functions.create_net_file import create_net_file
from utils_functions.compute_similarity import cosineSimilarity , euclideanSimilarity,\
    jensenshannonSimilarity, innerProductSimilarity, manhattanSimilarity,\
    softcosineSimliarity,WMDSimilarity
from utils_functions.pairs_list import create_pairs_file
from utils_functions.my_closeness_centerality import my_closeness_centrality
from utils_functions.my_betweenness_centrality import betweenness_centrality_parallel
from utils_functions.community_detections import Louvain_modularity_community, label_propagation_community ,\
    infomap_community, asyn_fluidc_community, k_clique_community
from utils_functions.clustering import average_linkage,kmeans, lda, ptm, nmf
import re
import numpy as np
from multiprocessing import Manager , Process

def load_data_forsim(datapath,name):
    if name in ['tf','tfidf'] :
        data = load_data(datapath).toarray().tolist()
    elif name in ['terms-tf','terms-tfidf'] :
        # data = [x for x in load_data(datapath)['tokens'].values]
        data = load_data(datapath)
    else:
        data = load_data(datapath)[0]
    return data

def compute_ths(sims_df,sims,manual_th,num_steps,th_map,rep):
    if not manual_th:
        mi = np.min(sims)
        ma = np.max(np.max(sims_df[sims_df !=1.0]))
        ths = []
        step_len = (ma - mi)/num_steps
        for i in range(num_steps):
            if mi +step_len*i <= ma:
                ths.append(round(mi +step_len*i,2) )
            else:
                break
    else:
        ths = [th_map[rep]]
    return ths

def compute_ks(sims_df,sims,manual_th,num_steps):
    step = int(len(sims_df)/num_steps)
    if not manual_th:
        ks = list(range(len(sims_df),0,-step))
    else:
        ks = 10
    return ks

def compute_similarities(similarity_func,datapath,name,manual_th,num_steps,th_map,rep):
    simsfile = datapath.split("/")[-1][:-4] + f"_{similarity_func.__name__}.pkl"
    simspath = os.path.join(datapath,"../",simsfile)
    if simsfile in os.listdir(os.path.join(datapath,"../")):
        sims_df,sims = load_data(simspath)
        ths = compute_ths(sims_df,sims,manual_th,num_steps,th_map,rep)
        ks = compute_ks(sims_df,sims,manual_th,num_steps)
        return ths,ks,sims_df
    data = load_data_forsim(datapath,name)
    sims_df,sims = similarity_func(data,name)
    ths = compute_ths(sims_df,sims,manual_th,num_steps,th_map,rep)
    ks = compute_ks(sims_df,sims,manual_th,num_steps)
    save_data(simspath,[sims_df,sims])
    return ths,ks,sims_df

def create_network_phase(name,network_func,network_arg,sims_df,savepath):
    print(name,"  ")
    # savepath = re.sub(savepath.split("_")[-1],str(threshold)+".pkl",savepath)
    g = network_func(sims_df,network_arg)
    print("edges: ",len(g.edges()),"  nodes: ",len(g.nodes()))
    if len(g.nodes)< int(len(sims_df)*0.8):
        return g
    return g

def centrality_phase(g,betweenness_processes,path,savepath):
    # _,g = load_data(savepath)
    print("edges: ",len(g.edges()),"  nodes: ",len(g.nodes()))
    funcList=[nx.degree_centrality]
    # funcList=[nx.degree_centrality, nx.pagerank,
    #           nx.eigenvector_centrality_numpy, my_closeness_centrality]
    return_dict = centrality_multi_process(funcList,[g])
    # return_dict = one_processing_func(g,betweenness_centrality_parallel,return_dict,processes=betweenness_processes)
    return_dict = modify(g,return_dict,path)
    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
    return return_dict,g

def community_detection_phase(community_models,return_dict,g,savepath):
    # return_dict,g= load_data(savepath)
    print("edges: ",len(g.edges()),"  nodes: ",len(g.nodes()))
    num_cl = len( set([y for x in pd.DataFrame(return_dict).transpose()['class'].unique() for y in str(x).split(",")]))
    # num_cl = len(pd.DataFrame(return_dict).transpose()['class'].unique())
    # try:
    for community_model in community_models:
        return_dict= community_model(g,return_dict,num_cl)
    # except:
    #     print(f"error in {community_model}")
    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
    # create_net_file(g,return_dict,savepath)
    return return_dict, g

def clustering_phase(clustering_models,return_dict,g,datapath,name,savepath):
    data = load_data_forsim(datapath,name)
    # num_cl = len(pd.DataFrame(return_dict).transpose()['class'].unique())
    num_cl = len( set([y for x in pd.DataFrame(return_dict).transpose()['class'].unique() for y in str(x).split(",")]))

    for clustering_model in clustering_models:
        if clustering_model.__name__== "lda" and f"LDA{num_cl}" not in name: continue
        if clustering_model.__name__== "ptm" and f"PTM{num_cl}" not in name: continue
        if clustering_model.__name__== "nmf" and name not in ['tf','tfidf']: continue
        return_dict = clustering_model(data,return_dict,num_cl)
    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")

def update_return_dict(return_dict,partition,modelname):
    for k,v in partition.items():
        return_dict[k][modelname] = v
    return  return_dict

def commuity_and_clustering_multiprocess_func(model,clustering_models,community_models,data,g,dict,num_cl):
    t1 = tm()
    if model in clustering_models:
        partition = model(data,return_dict = None,num_cl=num_cl)
    elif model in community_models:
        partition = model(g,return_dict = None,num_cl=num_cl)
    dict[model.__name__] = partition
    print(model.__name__,f" took {tm()-t1} s")

def commuity_and_clustering_multiprocess(clustering_models,community_models,return_dict,g,datapath,name,savepath,model_map):
    models = clustering_models + community_models
    print("edges: ",len(g.edges()),"  nodes: ",len(g.nodes()))
    data = load_data_forsim(datapath,name)
    # num_cl = len(pd.DataFrame(return_dict).transpose()['class'].unique())
    num_cl = len( set([y for x in pd.DataFrame(return_dict).transpose()['class'].unique() for y in str(x).split(",")]))
    dict = Manager().dict()
    processess= []
    for c in range(len(models)):
        if models[c].__name__== "lda" and f"LDA{num_cl}" not in name: continue
        if models[c].__name__== "ptm" and f"PTM{num_cl}" not in name: continue
        if models[c].__name__== "nmf" and name not in ['tf','tfidf']: continue
        p = Process(target = commuity_and_clustering_multiprocess_func, args=(models[c],clustering_models,community_models,data,g,dict,num_cl) )
        p.start()
        processess.append(p)

    for p in processess:
        p.join()

    for m in dict.keys():
        partition = dict[m]
        modelname = model_map[m]
        return_dict = update_return_dict(return_dict,partition,modelname)

    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")


if __name__ == "__main__":
    print(tm())
    paramspath = f"text_clustering_params.xlsx"
    manual_th = True
    keep_nodes = 1
    num_steps = 20
    network_func = [knn,enn][1]
    # similarity_functions = [innerProductSimilarity, cosineSimilarity , euclideanSimilarity,
    #                         jensenshannonSimilarity, manhattanSimilarity,softcosineSimliarity,
    #                         # WMDSimilarity
    #                         ]
    similarity_functions = [cosineSimilarity
                            ]

    # num_steps = 1
    # similarity_functions = [innerProductSimilarity]

    community_models = [Louvain_modularity_community, label_propagation_community,
                        infomap_community, asyn_fluidc_community,
                        # k_clique_community
                        ]

    clustering_models = [average_linkage, kmeans, lda, ptm,
                         # nmf
                         ]
    model_map = {
        Louvain_modularity_community.__name__:"Louvain_modularity",
        label_propagation_community.__name__:"label_propagation",
        infomap_community.__name__:"infomap",
        asyn_fluidc_community.__name__:"asyn_fluidc",
        average_linkage.__name__:"average_linkage",
        kmeans.__name__:"kmeans",
        lda.__name__:"lda",
        ptm.__name__:"ptm",
    }
    th_map={
        'use':0.6,
        'bert':0.6,
        'tf':0.04,
        'tfidf':0.04,
        'elmo-default':0.6,
        'PTM3':0.94,
        'PTM20':0.6,
        'PTM50':0.3,
        'PTM100':0.2,
        'LDA3':0.84,
        'LDA20':0.3,
        'LDA50':0.3,
        'LDA100':0.3,
    }
    paramsdf = pd.read_excel(paramspath)
    prev_dataset = None
    for iter in range(len(paramsdf)):
        # threshold = paramsdf['threshold'][iter]
        name = paramsdf['name'][iter]
        data_version = paramsdf['data_version'][iter]
        dataset = paramsdf['dataset'][iter]
        betweenness_processes = paramsdf['betweenness_processes'][iter]

        datapath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{name}.pkl"
        path = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"


        for similarity_func in similarity_functions:
            if 'terms' not in name  and similarity_func.__name__ in ["softcosineSimliarity","WMDSimilarity"]:
                continue
            if 'LDA' not in name and similarity_func.__name__ in ["jensenshannonSimilarity"]:
                continue
            if 'terms' in name  and similarity_func.__name__ not in ["softcosineSimliarity","WMDSimilarity"]:
                continue
            print(f"\n  compute_similarities _ {similarity_func.__name__}_ {name} _\ndatapath {datapath}" )
            # if dataset != prev_dataset:
            #     ths ,sims_df = compute_similarities(similarity_func,datapath,name,manual_th,num_steps)
            #     prev_dataset = dataset
            ths,ks ,sims_df = compute_similarities(similarity_func,datapath,name,manual_th,num_steps,th_map,name)

            if network_func.__name__=="enn":
                network_args = ths
            else:
                network_args = ks

            for network_arg in network_args:
                # savepath = f"data/output/{dataset}/{data_version}/network/{dataset}_preprocessed_data_{data_version}_{name}_network_th_{threshold}.pkl"
                savepath = f"data/output/{dataset}/{data_version}/network/{dataset}_preprocessed_data_{data_version}_{name}_network_{network_func.__name__}_{network_arg}_{similarity_func.__name__}.pkl"
                if "network" not in os.listdir("/".join(savepath.split("/")[:4])):
                    os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"network"))
                print(f"\n\n  create_network _ {network_arg} _ {name}" )
                # g , threshold = create_network_phase(name,threshold,sims_df,savepath)
                g  = create_network_phase(name,network_func,network_arg,sims_df,savepath)
                if len(g.nodes)< int(len(sims_df)*keep_nodes):
                    break
                else:
                    create_pairs_file(g,savepath[:-4]+".pairs")
                    save_data(savepath,[None,g])
                print(f"  centrality _ {network_arg} _ {name} " )
                return_dict,g = centrality_phase(g,betweenness_processes,path,savepath)

                t1=tm()
                print(f"  community_detection _ {network_arg} _ {name} " )
                return_dict, g = community_detection_phase(community_models,return_dict,g,savepath)

                if name in ['terms-tf']:
                    continue
                print(f"  clustering _ {name} " )
                clustering_phase(clustering_models,return_dict,g,datapath,name,savepath)
                print(tm()-t1)

                # t1=tm()
                # print(f"  community_detection and clustering _ {network_arg} _ {name} " )
                # commuity_and_clustering_multiprocess(clustering_models,community_models,return_dict,g,datapath,name,savepath,model_map)
                # print(tm()-t1)

