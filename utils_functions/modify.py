from utils_functions.utils import *


# def modify(g,return_dict,datapath,oneOrTotal="total"):
#     # data = load_data("data/preprocessed_data(polarity_added).pkl",article=oneOrTotal)
#     data = load_data(datapath,article=oneOrTotal)
#     data = data[0]
#     data_dict={}
#     for n in list(g.nodes()): data_dict[n]={}
#
#     for col in data.columns:
#         for n in g.nodes():
#             data_dict[n][col]=data[col][data["commentID"]==n].values[0]
#
#     for i in return_dict.keys():
#         for k,v in return_dict[i].items():
#             data_dict[k][i]=v
#     return data_dict

def modify(g,return_dict,path):
    data = load_data(path)
    data_dict={}
    for n in list(g.nodes()): data_dict[n]={}
    for i in return_dict.keys():
        for k,v in return_dict[i].items():
            data_dict[k][i]=v
            data_dict[k]["text"]= data["text"][k]
            data_dict[k]["tokens"]= data["tokens"][k]
            data_dict[k]["class"]= data["class"][k]
            data_dict[k]["class_label"]= data["class_label"][k]
    return data_dict
