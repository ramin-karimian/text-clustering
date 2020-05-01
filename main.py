import networkx as nx
from time import time as tm
import os
from utils_functions.utils import *
from utils_functions.create_network import create_network
from utils_functions.modify import modify
from utils_functions.multi_processing import centrality_multi_process , network_multi_process , one_processing_func
from utils_functions.create_net_file import create_net_file
from utils_functions.compute_similarity import similarity
from utils_functions.my_closeness_centerality import my_closeness_centrality
from utils_functions.my_betweenness_centrality import betweenness_centrality_parallel
from utils_functions.community_detections import Louvain_modularity_community as community_detection


def create_network_phase(name,threshold,datapath,savepath):
    print(name,"  ",threshold)
    if name in ['tf','tfidf'] : data = load_data(datapath).toarray().tolist()
    else:  data = load_data(datapath)[0]
    sims_df,sims = similarity(data)
    g = create_network(sims_df,threshold)
    save_data(savepath,[None,g])
    print("edges: ",len(g.edges()))
    print("nodes: ",len(g.nodes()))
    return g

def centrality_phase(g,betweenness_processes,path):
    # _,g = load_data(savepath)
    print("edges: ",len(g.edges()))
    print("nodes: ",len(g.nodes()))
    funcList=[nx.degree_centrality]
    # funcList=[nx.degree_centrality, nx.pagerank,
    #           nx.eigenvector_centrality_numpy, my_closeness_centrality]
    return_dict = centrality_multi_process(funcList,[g])
    # return_dict = one_processing_func(g,betweenness_centrality_parallel,return_dict,processes=betweenness_processes)
    return_dict = modify(g,return_dict,path)
    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
    return return_dict,g

def community_detection_phase(return_dict,g):
    # return_dict,g= load_data(savepath)
    print("edges: ",len(g.edges()))
    print("nodes: ",len(g.nodes()))
    return_dict= community_detection(g,return_dict)
    save_data(savepath,[return_dict,g])
    pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
    # create_net_file(g,return_dict,savepath)


if __name__ == "__main__":
    print(tm())
    paramspath = f"text_clustering_params.xlsx"
    paramsdf = pd.read_excel(paramspath)
    for iter in range(len(paramsdf)):
        threshold = paramsdf['threshold'][iter]
        name = paramsdf['name'][iter]
        data_version = paramsdf['data_version'][iter]
        dataset = paramsdf['dataset'][iter]
        betweenness_processes = paramsdf['betweenness_processes'][iter]

        datapath = f"data/output/{dataset}/{data_version}/representation/{dataset}_preprocessed_data_{data_version}_{name}.pkl"
        savepath = f"data/output/{dataset}/{data_version}/network/{dataset}_preprocessed_data_{data_version}_{name}_network_th_{threshold}.pkl"
        path = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"

        if "network" not in os.listdir("/".join(savepath.split("/")[:4])):
            os.mkdir(os.path.join("/".join(savepath.split("/")[:4]),"network"))

        print(f"\ncreate_network _ {threshold} _ {name} _\ndatapath {datapath}\nsavepath {savepath}" )
        g = create_network_phase(name,threshold,datapath,savepath)

        print(f"\ncentrality _ {threshold} _ {name} _\ndatapath {datapath}\nsavepath {savepath}" )
        return_dict,g = centrality_phase(g,betweenness_processes,path)

        print(f"\ncommunity_detection _ {threshold} _ {name} _\ndatapath {datapath}\nsavepath {savepath}" )
        community_detection_phase(return_dict,g)



