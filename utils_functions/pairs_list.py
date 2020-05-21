from utils_functions.utils import *


def create_pairs_file(g,savepath):
    edges = list(g.edges)
    with open(savepath,'w') as f:
        for e in edges:
            f.write(f"{e[0]},{e[1]},{round(g[e[0]][e[1]]['weight'],2)}\n")

if __name__=="__main__":
    datapath = f"../data/output/bbcsport/V02/network/bbcsport_preprocessed_data_V02_use_network_th_0.54.pkl"
    savepath = f"bbcsport_use_network_th_0.54.pairs"

    _,g = load_data(datapath)
    edges = list(g.edges)
    with open(savepath,'w') as f:
        for e in edges:
            f.write(f"{e[0]},{e[1]},{round(g[e[0]][e[1]]['weight'],2)}\n")

