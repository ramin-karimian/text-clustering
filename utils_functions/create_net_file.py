import networkx as nx

def create_net_file(g,return_dict,savepath):
    for i,k in return_dict.items():
        for ii,kk in k.items():
            if ii=="emb":continue
            g.nodes[i][ii]=kk
    nx.write_gexf(g,savepath[:-4]+".gexf")
