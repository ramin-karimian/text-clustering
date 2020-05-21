import os
from utils_functions.utils import *
from data.scripts.partition_density import partition_density_from_file


if __name__=="__main__":
    # dataset="twitter-cor2"
    dataset="bbcsport"
    data_version="V02"
    path = f"../output/{dataset}/{data_version}/network/linked_community_results"
    respath = f"../output/{dataset}/{data_version}/network/linked_community_res.xlsx"
    ups = [1,5,10]
    df = pd.DataFrame(columns=['filename','edgesnum','max_cl_size','total_cl_num',f'up{ups[0]}_cl_num',f'up{ups[1]}_cl_num',f'up{ups[2]}_cl_num','D(part. dens.)'])
    for filename in os.listdir(path):
        if filename.endswith(".comm2nodes"):
            D = partition_density_from_file(f"{path}/{filename[:-11]}.mc_nc")
            fname= filename.split("_")
            print(filename)

            f1 = open(f"{path}/{filename[:-11]}.clusters","r")
            lns = f1.readlines()
            lns = [l[:-2].split(" ") for l in lns ]
            edgesnum = sum([len(x) for x in lns])
            f1.close()

            # filepath = f"{path}/{filename}"
            # f = open(filepath,"r")
            # lns = f.readlines()
            # lns = [l[:-1].split("\t") for l in lns ]
            # nodesnum = sum([len(x) for x in lns])
            df.loc[len(df)] = [f"{fname[-5]}({fname[-2]})_{fname[-1].split('.c')[0]}",
                               edgesnum,
                               max([len(x) for x in lns ]),
                               len(lns),
                               len([x for x in lns if len(x)>ups[0]]),
                               len([x for x in lns if len(x)>ups[1]]),
                               len([x for x in lns if len(x)>ups[2]]),
                               D
                               ]
            # f.close()
    df.to_excel(respath)
