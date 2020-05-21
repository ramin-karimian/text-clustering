from time import time as tm
import subprocess
import os
from utils_functions.link_community_detection_cpp_utils import *
from utils_functions.comm2nodes import write_comm2nodes

if __name__ == '__main__':
    conf = {
        # 'dataset' : "twitter-cor2",
        # 'dataset' : "bbcsport",
        'dataset' : "bbcsport-small",
        # 'dataset' : "twitter",
        # 'dataset' : "cran_cisi_pubmed",
        # 'data_version' : "V02" ,
        'data_version' : "V01" ,
        'path' : f"data/output",
        "compute_jaccards":True,
        "threshold":True,
        "is_weighted":False,
        "dendro_flag":False,
        "cores":12,
        "delimiter":",",
        "th_list":[0.1,0.3,0.5,0.7],
        # "th_list":[0.2,0.25,0.3,0.35,0.4],
        # "th_list":[0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7],
        # "th_list":[0.5,0.2,0.15,0.1,0.05],
        # "th_list":[0.5,0.1],
        'clusterJaccards_program':'utils_functions/clusterJaccards',
        # 'calcJaccards_program':'utils_functions/calcJaccards_weighted.exe',
    }
    if conf['is_weighted']: conf['calcJaccards_program'] = 'utils_functions/calcJaccards_weighted.exe'
    else: conf['calcJaccards_program'] = 'utils_functions/calcJaccards_unweighted.exe'
    folderpath = os.path.join(conf['path'],conf['dataset'],conf['data_version'],'network')
    for filename in os.listdir(folderpath):
        if filename.endswith(".pairs"):
            conf["pairsfilepath"] = os.path.join(folderpath,filename)
            conf["resultspath"] = os.path.join(folderpath,'linked_community_results')
            if 'linked_community_results' not in os.listdir(folderpath): os.mkdir(conf['resultspath'])

            t1 = tm()
            if conf["delimiter"] == '\\t':
                conf["delimiter"] = '\t'

            if conf["compute_jaccards"]:
                print("converting pairs file")
                convert_pairs_file(conf,filename)
                t2 = print_time(t1)

                print("computing jaccards")
                compute_jaccards_cpp_integrated(conf,filename)
                t3 = print_time(t2)

                print("processing jaccards' files")
                process_jaccards_files_cpp(conf['resultspath'],filename)
                t4 = print_time(t3)

            print("running jaccs clustering")
            for th in conf["th_list"]:
                tt = tm()
                res = run_jaccs_clustering(conf,th,filename)
                write_comm2nodes(conf,filename,th)
                t5 = print_time(tt)

# if __name__ == '__main__':
#     conf = {
#         "compute_jaccards":True,
#         "pairsfilepath":"bbcsport_use_network_th_0.54/bbcsport_use_network_th_0.54.pairs",
#         "threshold":False,
#         "is_weighted":True,
#         "dendro_flag":False,
#         "cores":14,
#         "delimiter":",",
#         # "th_list":[0.1,0.5,0.8,0.95],
#         # "th_list":[0.2,0.25,0.3,0.35,0.4],
#         # "th_list":[0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7],
#         "th_list":[0.5,0.15,0.2,0.1],
#         'clusterJaccards_program':'clusterJaccards',
#         'calcJaccards_program':'calcJaccards_weighted.exe',
#     }
#     conf["resultspath"] = conf["pairsfilepath"].split("/")[0]
#     t1 = tm()
#
#     if conf["delimiter"] == '\\t':
#         conf["delimiter"] = '\t'
#
#     if conf['resultspath'] not in os.listdir(): os.mkdir(conf['resultspath'])
#
#     if conf["compute_jaccards"]:
#         print("converting pairs file")
#         convert_pairs_file(conf)
#         t2 = print_time(t1)
#
#         print("computing jaccards")
#         compute_jaccards_cpp_integrated(conf)
#         t3 = print_time(t2)
#
#         print("processing jaccards' files")
#         process_jaccards_files_cpp(conf['resultspath'])
#         t4 = print_time(t3)
#
#     print("running jaccs clustering")
#     for th in conf["th_list"]:
#         tt = tm()
#         res = run_jaccs_clustering(conf,th)
#         t5 = print_time(tt)
#
