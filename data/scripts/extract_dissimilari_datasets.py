from utils_functions.compute_similarity import innerProductSimilarity
from utils_functions.utils import *
from main import load_data_forsim

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

if __name__=="__main__":
    # datapath = f"../output/bbcsport/V02/representation/bbcsport_preprocessed_data_V02_tf.pkl"
    datapath = f"../output/cran-cisi-pubmed/V02/representation/cran-cisi-pubmed_preprocessed_data_V02_tf.pkl"
    savepath = f"../mixed_preprocessed_data_V02.pkl"
    datasetpath = f"../output/cran-cisi-pubmed/V02/cran-cisi-pubmed_preprocessed_data_V02.pkl"
    df = load_data(datasetpath)
    ranges = [(1,1397),(1398,2856),(2857,6300)]
    new_df = pd.DataFrame(columns= df.columns)

    data = load_data_forsim(datapath,name='tf')
    sims_df,sims = innerProductSimilarity(data,name='tf')

    new_df = pd.concat([new_df ,df.loc[:33]])
    new_df = pd.concat([new_df ,df.loc[:0]])
    rmvlist = []
    rg = ranges[0]
    inds = sims_df[0][rg[0]:rg[1]].argsort()[::-1].values +rg[0]
    cc = 0
    c = 0
    while c < 32:
        ind = inds[cc]
        if  100 < len(df.loc[ind]['tokens']) or len(df.loc[ind]['tokens']) < 60:
            cc = cc+1
            # c = c-1
            continue
        new_df.loc[len(new_df)] = df.loc[ind]
        rmvlist.append(ind)
        cc = cc+1
        c = c+1

    rg = ranges[1]
    ln = len(new_df)
    inds = sims_df[0][rg[0]:rg[1]].argsort().values +rg[0]
    cc = 0
    c = 0
    # for c in range(ln):
    while c < 32:
        ind = inds[cc]
        if 100 < len(df.loc[ind]['tokens']) or len(df.loc[ind]['tokens']) < 60:
            cc = cc+1
            # c = c-1
            continue
        new_df.loc[len(new_df)] = df.loc[ind]
        rmvlist.append(ind)
        cc = cc+1
        c = c+1

    rg = ranges[2]
    set1 = sims_df[0][rg[0]:rg[1]].argsort().values +rg[0]
    set2 = sims_df[ln][rg[0]:rg[1]].argsort().values +rg[0]
    inds = intersection(set1,set2)
    cc = 0
    c = 0
    # for c in range(ln,ln + 32):
    while c < 32:
        ind = inds[cc]
        if 100 < len(df.loc[ind]['tokens']) or len(df.loc[ind]['tokens']) < 60:
            cc = cc+1
            # c = c-1
            continue
        new_df.loc[len(new_df)] = df.loc[ind]
        rmvlist.append(ind)
        cc = cc+1
        c = c+1
    save_data(savepath,new_df)
    new_df.to_excel(savepath[:-4]+".xlsx")
        # new_df.loc[len(new_df)] = df.loc[ind]
