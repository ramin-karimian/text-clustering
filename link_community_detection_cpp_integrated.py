from time import time as tm
import subprocess
import os
from utils_functions.link_community_detection_cpp_utils import *
from utils_functions.utils import *


def compute_linkcom_total(conf,lim,sim='cosineSimilarity', mode = ''):

    folderpath = os.path.join(conf['path'],conf['dataset'],conf['data_version'],'network')
    l = len(os.listdir(folderpath))
    for c , filename in enumerate(os.listdir(folderpath)):
        if filename.endswith(".pairs"):

            if mode == "knn":
                if filename.split("_")[6] != mode or sim not in filename.split("_")[8] or float(filename.split("_")[7]) >= lim :
                    continue
            if mode == "enn":
                # if filename.split("_")[6] != mode or sim not in filename.split("_")[8] or float(filename.split("_")[7]) <= lim :
                if filename.split("_")[6] != mode or sim not in filename.split("_")[8] or float(filename.split("_")[7]) > lim :
                    continue

            conf["pairsfilepath"] = os.path.join(folderpath,filename)
            conf["resultspath"] = os.path.join(folderpath,'linked_community_results')
            if 'linked_community_results' not in os.listdir(folderpath): os.mkdir(conf['resultspath'])

            savepath = os.path.join(folderpath,filename)[:-6]+".pkl"
            return_dict, g = load_data(savepath)
            if return_dict is None: continue
            t1 = tm()
            if conf["delimiter"] == '\\t':
                conf["delimiter"] = '\t'

            if conf["compute_jaccards"]:
                print(f"converting pairs file _ {c}/{l}")
                print(filename)
                convert_pairs_file(conf,filename)
                # t2 = print_time(t1)

                print("computing jaccards")
                compute_jaccards_cpp_integrated(conf,filename)
                # t3 = print_time(t2)

                print("processing jaccards' files")
                process_jaccards_files_cpp(conf['resultspath'],filename)
                # multiprocess_jaccards_files_cpp(conf['resultspath'],filename)
                # t4 = print_time(t3)

            print("running jaccs clustering")
            return_dict = jaccs_clustering_multiprocess(conf,filename,return_dict)
            save_data(savepath,[return_dict,g])
            pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")

def compute_linkcom_incase(conf,rep,mode,sim='',lim=0):
    folderpath = os.path.join(conf['path'],conf['dataset'],conf['data_version'],'network')
    l = len(os.listdir(folderpath))
    # l = len(os.listdir(folderpath))
    # print(len(os.listdir(folderpath)))
    # print(os.listdir(folderpath))
    for c , filename in enumerate(os.listdir(folderpath)):
        if filename.endswith(".pairs"):
            # if filename.split("_")[4] != rep or sim not in filename.split("_")[8]  :
            #     continue
            if mode == "knn":
                if filename.split("_")[4] != rep or sim not in filename.split("_")[8] or filename.split("_")[6] != mode or float(filename.split("_")[7]) != lim :
                    continue
            if mode == "enn":
                if filename.split("_")[4] != rep or sim not in filename.split("_")[8] or filename.split("_")[6] != mode or float(filename.split("_")[7]) != lim :
                    continue
            conf["pairsfilepath"] = os.path.join(folderpath,filename)
            conf["resultspath"] = os.path.join(folderpath,'linked_community_results')
            if 'linked_community_results' not in os.listdir(folderpath): os.mkdir(conf['resultspath'])

            savepath = os.path.join(folderpath,filename)[:-6]+".pkl"
            return_dict, g = load_data(savepath)
            print(f'hi {filename.split("_")[4]}')
            if return_dict is None: continue
            t1 = tm()
            if conf["delimiter"] == '\\t':
                conf["delimiter"] = '\t'

            if conf["compute_jaccards"]:
                # print(f"converting pairs file _ {c}/{l}")
                # print(filename)
                # convert_pairs_file(conf,filename)
                # # t2 = print_time(t1)
                #
                # print("computing jaccards")
                # compute_jaccards_cpp_integrated(conf,filename)
                # # t3 = print_time(t2)

                print("processing jaccards' files")
                process_jaccards_files_cpp(conf['resultspath'],filename)
                # multiprocess_jaccards_files_cpp(conf['resultspath'],filename)
                # t4 = print_time(t3)

            print("running jaccs clustering")
            return_dict = jaccs_clustering_multiprocess(conf,filename,return_dict)
            save_data(savepath,[return_dict,g])
            pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")
            # break
    # return return_dict,g


if __name__ == '__main__':
    conf = {
        # 'dataset' : "twitter-cor2",
        # 'dataset' : "bbcsport",
        # 'dataset' : "bbcsport-small",
        # 'dataset' : "mixed",
        # 'dataset' : "twitter",
        # 'dataset' : "mixedOverlap",
        # 'dataset' : "cran_cisi_pubmed",
            # dataset = "cran-cisi-pubmed-1000"
        # 'dataset' : "cran-cisi-pubmed-1000-overlapped",
        'dataset' : "cran-cisi-pubmed-overlapped",
        # 'dataset' : "cran-cisi-pubmed-100",
        # 'dataset' : "cran-cisi-pubmed-300",
        # 'dataset' : "cran-cisi-pubmed-300-overlapped",
        # 'dataset' : "cran-cisi-pubmed-100-overlapped",
        'data_version' : "V02" ,
        # 'data_version' : "V01" ,
        'path' : f"data/output",
        "compute_jaccards":True,
        "threshold":True,
        "is_weighted":False,
        "dendro_flag":False,
        "delete_jacfile":False,
        "cores":15,
        "delimiter":",",
        "th_list":[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8],
        # "th_list":[0.2,0.25,0.3,0.35,0.4],
        # "th_list":[0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7],
        # "th_list":[0.5,0.2,0.15,0.1,0.05],
        # "th_list":[0.5,0.1],
        'clusterJaccards_program':'utils_functions/clusterJaccards',
        # 'calcJaccards_program':'utils_functions/calcJaccards_weighted.exe',
    }
    if conf['is_weighted']: conf['calcJaccards_program'] = 'utils_functions/calcJaccards_weighted.exe'
    else: conf['calcJaccards_program'] = 'utils_functions/calcJaccards_unweighted.exe'
    # compute_linkcom_incase(conf,rep='elmo-default',mode = 'enn',sim='',lim=-1)
    # compute_linkcom_incase(conf,rep='elmo-default',mode = 'knn',sim='',lim=10000)
    compute_linkcom_incase(conf,rep='LDA3',mode = 'enn',sim='cosineSimilarity',lim=0.84)
    # compute_linkcom_total(conf,lim=0.15,mode = 'enn') #1000-over
    # compute_linkcom_total(conf,lim=0, sim = 'cosineSimilarity', mode = 'enn') #300





    # folderpath = os.path.join(conf['path'],conf['dataset'],conf['data_version'],'network')
    # l = len(os.listdir(folderpath))
    # for c , filename in enumerate(os.listdir(folderpath)):pythoon --v

    #     if filename.endswith(".pairs"):
    #
    #         conf["pairsfilepath"] = os.path.join(folderpath,filename)
    #         conf["resultspath"] = os.path.join(folderpath,'linked_community_results')
    #         if 'linked_community_results' not in os.listdir(folderpath): os.mkdir(conf['resultspath'])
    #
    #         savepath = os.path.join(folderpath,filename)[:-6]+".pkl"
    #         return_dict, g = load_data(savepath)
    #         if return_dict is None: continue
    #         t1 = tm()
    #         if conf["delimiter"] == '\\t':
    #             conf["delimiter"] = '\t'
    #
    #         if conf["compute_jaccards"]:
    #             print(f"converting pairs file _ {c}/{l}")
    #             print(filename)
    #             convert_pairs_file(conf,filename)
    #             # t2 = print_time(t1)
    #
    #             print("computing jaccards")
    #             compute_jaccards_cpp_integrated(conf,filename)
    #             # t3 = print_time(t2)
    #
    #             print("processing jaccards' files")
    #             process_jaccards_files_cpp(conf['resultspath'],filename)
    #             # t4 = print_time(t3)
    #
    #         print("running jaccs clustering")
    #
    #         return_dict = jaccs_clustering_multiprocess(conf,filename,return_dict)
    #
    #         # break
    #         save_data(savepath,[return_dict,g])
    #         pd.DataFrame(return_dict).transpose().to_excel(savepath[:-4]+".xlsx")

