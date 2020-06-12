import pandas as pd
import networkx as nx
from utils_functions.utils import *
from utils_functions.compute_similarity import innerProductSimilarity, cosineSimilarity
import community
from sklearn.metrics.pairwise import cosine_similarity


def create_net(sims_df):
    g = nx.Graph()
    for i in range(len(sims_df)):
        for j in range( i +1 , len(sims_df)):
            if sims_df[i][j] > 0:
                g.add_edge(i,j,weight=sims_df[i][j])
    return g

def detect_communities(g):
    partition = community.best_partition(g)
    return partition

def read_topicsfile(topicspath):
    with open(topicspath,'r') as f:
        lns = f.readlines()
        assert len(df) == len(lns) , f"data len error, check ids "
        ids = []
        data = []
        for x in lns:
            ids.append(len(ids))
            l = []
            for y in x[:-2].split(" "):
                l.append(float(y))
            data.append(l)
    return data

def create_report_community(partition,df):
    num_part = len(set(partition.values()))
    num_cl = len( set([y for x in df['class'].unique() for y in str(x).split(",")]))
    repo = {}
    for c in range(num_cl):
        print(c)
        # l = [0] * num_part
        for i in range(len(df)):
            if str(c) in str(df['class'][i]):
                if str(c) not in repo:
                    repo[str(c)] = [0] * num_part
                inds = [(j,n) for j, n in enumerate(rep[i]) if n >= 1]
                for j,n in inds:
                    if j not in partition: continue
                    repo[str(c)][partition[j]] += n
    repolist = []
    for r in repo:
        repo[r] = np.around(np.divide(repo[r],sum(repo[r])),2)
        repolist.append(repo[r])
    return repo , repolist

def create_report_topical(df,ntopics):
    num_part = ntopics
    num_cl = len( set([y for x in df['class'].unique() for y in str(x).split(",")]))
    repo = {}
    repolist=[]
    for c in range(num_cl):
        print(c)
        counter = 0
        for i in range(len(df)):
            if str(c) in str(df['class'][i]):
                if str(c) not in repo:
                    repo[str(c)] = [0] * num_part
                counter +=1
                repo[str(c)] = np.sum([repo[str(c)],data[i]],axis=0)
        repo[str(c)] = np.around(np.divide(repo[str(c)],counter),2)
        repolist.append(repo[str(c)])
    return repo, repolist

def compute_similarity(repolist):
    sims = cosine_similarity([x for x in repolist],[x for x in repolist])
    sims = np.around(sims.tolist(),2)
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    avgsim = np.sum(np.sum(sims_df[sims_df<1]))/ (len(sims_df)**2 - len(sims_df))
    return sims_df , round(avgsim,2)

if __name__=="__main__":
    modelname = 'PTM'
    ntopics = 20
    model_version ="3"
    # ntopics = 3
    # model_version ="4"
    # ntopics = 50
    # model_version ="2"

    # dataset = "cran-cisi-pubmed-1000-overlapped"
    dataset = "cran-cisi-pubmed-1000"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    data_version = "V02"
    topicspath = f"C:/Users/RAKA/Documents/NLP/topic modeling/STTM-master_java/results/{modelname}_{dataset}_{data_version}_T{ntopics}_{model_version}.theta"
    datapath = f"data/output/{dataset}/{data_version}/{dataset}_preprocessed_data_{data_version}.pkl"
    df = load_data(datapath)

    data = read_topicsfile(topicspath)

    repo, repolist = create_report_topical(df,ntopics)

    sims_df , avgsim = compute_similarity(repolist)
    print(sims_df , avgsim)



if __name__=="__main__2":
    # reppath = f"C:/Users/RAKA/Documents/text clustring/data/output/cran-cisi-pubmed-300-overlapped/V02/representation/cran-cisi-pubmed-300-overlapped_preprocessed_data_V02_tf.pkl"
    # datapath = f"C:/Users/RAKA/Documents/text clustring/data/output/cran-cisi-pubmed-300-overlapped/V02/cran-cisi-pubmed-300-overlapped_preprocessed_data_V02.pkl"
    reppath = f"C:/Users/RAKA/Documents/text clustring/data/output/cran-cisi-pubmed-1000-overlapped/V02/representation/cran-cisi-pubmed-1000-overlapped_preprocessed_data_V02_tf.pkl"
    datapath = f"C:/Users/RAKA/Documents/text clustring/data/output/cran-cisi-pubmed-1000-overlapped/V02/cran-cisi-pubmed-1000-overlapped_preprocessed_data_V02.pkl"
    df =  load_data(datapath)
    # df = pd.DataFrame(load_data(datapath).toarray().tolist())
    rep = load_data(reppath).toarray().tolist()
    wordsrep = (np.reshape(rep,(np.shape(rep)[1],np.shape(rep)[0])))
    print("rep")
    sims_df,sims = innerProductSimilarity(wordsrep,'tf')
    print("net")
    g = create_net(sims_df)

    partition = detect_communities(g)

    repo, repolist = create_report_community(partition,df)

    sims_df , avgsim = compute_similarity(repolist)
    print(sims_df , avgsim)

