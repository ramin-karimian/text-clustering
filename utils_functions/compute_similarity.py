import os
from sklearn.metrics.pairwise import cosine_similarity , euclidean_distances , manhattan_distances
from utils_functions.utils import *
import gensim.downloader as api
from scipy.spatial.distance import jensenshannon
import gensim
from gensim import corpora
from gensim.matutils import softcossim
import gensim.downloader as api
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix,  WmdSimilarity
from multiprocessing import Process,Manager

def normalize(sims,sims_df):
    # minimum= np.min(np.min(sims_df[sims_df !=0.0]))
    # maximum= np.max(np.max(sims_df[sims_df !=np.max(np.max(sims_df))]))
    maximum = np.max(sims)
    minimum = np.min(sims)
    sims = (sims -minimum )/( maximum - minimum )
    sims = np.around(sims.tolist(),4)
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    if sims.any() > 1 or  sims.any() <0:
        print("sims error")
    return sims,sims_df


def cosineSimilarity(data,name):
    sims = cosine_similarity([x for x in data],[x for x in data])
    sims = np.around(sims.tolist(),2)
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    return sims_df,sims

def euclideanSimilarity(data,name):
    sims = euclidean_distances([x for x in data],[x for x in data])
    sims = np.around(sims.tolist(),4)
    sims = 1 - sims
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    sims,sims_df = normalize(sims,sims_df)
    return sims_df,sims

def jensenshannonSimilarity(data,name):
    sims_df = pd.DataFrame(None,index = range(len(data)), columns= range(len(data)))
    for i in range(len(data)):
        check_print(i,step=100)
        for j in range(i,len(data)):
            sim = jensenshannon(data[i], data[j])
            sims_df.iloc[i][j] = sims_df.iloc[j][i] = sim
    sims = np.array(sims_df)
    sims = 1 - sims
    sims = sims.tolist()
    sims,sims_df = normalize(sims,sims_df)
    return sims_df,sims

def innerProductSimilarity(data,name):
    sims = np.inner(data , data)
    sims = np.around(sims.tolist(),4)
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    sims,sims_df = normalize(sims,sims_df)
    return sims_df,sims

def manhattanSimilarity(data,name):
    sims = manhattan_distances(data , data)
    sims = 1 - sims
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    sims,sims_df = normalize(sims,sims_df)
    return sims_df,sims

def softcosineSimliarity(data,name):
    # fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    if name.split("-")[1] =='tf':
        tfidf = None
    elif name.split("-")[1] =='tfidf':
        tfidf = gensim.models.tfidfmodel.TfidfModel
    else:
        return
    dictionary = corpora.Dictionary(data)
    model = gensim.models.KeyedVectors.load_word2vec_format ('E:/datasets/word_embedding/GoogleNews-vectors-negative300.bin', binary=True)
    termsim_index = WordEmbeddingSimilarityIndex(model.wv)
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary,tfidf= tfidf)
    bow_data = [dictionary.doc2bow(document) for document in data]
    docsim_index = SoftCosineSimilarity(bow_data, similarity_matrix)

    sims_df = pd.DataFrame(None,index = range(len(data)), columns= range(len(data)))
    for i in range(len(bow_data)):
        check_print(i,step=100)
        sims_df.iloc[i] = docsim_index[bow_data[i]]
    sims,sims_df = normalize(np.array(sims_df).tolist(),sims_df)
    return sims_df,sims

def WMDSimilarity(data,name):
    # fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    model = gensim.models.KeyedVectors.load_word2vec_format ('E:/datasets/word_embedding/GoogleNews-vectors-negative300.bin', binary=True)
    docsim_index = WmdSimilarity(data, model)

    sims_df = pd.DataFrame(None,index = range(len(data)), columns= range(len(data)))
    for i in range(len(data)):
        check_print(i,step=100)
        sims_df.iloc[i] = docsim_index[data[i]]
    sims,sims_df = normalize(np.array(sims_df).tolist(),sims_df)
    return sims_df,sims

# def WMD_similarity_func(docsim_index,core,return_dict,data):
#     sims = []
#     for i in range(len(data)):
#         check_print(i,message=f"for core {core+1} {i}/{len(data)}",step=10)
#         sims.append(docsim_index[data[i]].tolist())
#     return_dict[core] = sims
#
# def WMD_similarity(data,cores=10):
#     model = gensim.models.KeyedVectors.load_word2vec_format ('E:/datasets/word_embedding/GoogleNews-vectors-negative300.bin', binary=True)
#     sims_df = pd.DataFrame(None,index = range(len(data)), columns= range(len(data)))
#     docsim_index = WmdSimilarity(data, model)
#     return_dict = Manager().dict()
#     processes = []
#     step = int(len(data)/cores)
#     prev_end = 0
#     for c in range(cores):
#         start = prev_end
#         if c != cores-1:
#             end = start + step
#         else:
#             end = len(data)
#         prev_end = end
#         p = Process(target=WMD_similarity_func,args=(docsim_index,c,return_dict,data[start:end]))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
#     sims = []
#     for c in return_dict.keys():
#         sims.extend(return_dict[c])
#     sims = np.array(sims_df).tolist()
#     sims,sims_df = normalize(sims,sims_df)
#     return sims,sims_df

# def WMD_similarity(data):
#     # model = api.load('word2vec-google-news-300')
#     model = gensim.models.KeyedVectors.load_word2vec_format ('E:/datasets/word_embedding/GoogleNews-vectors-negative300.bin', binary=True)
#     sims_df = pd.DataFrame(None,index = range(len(data)), columns= range(len(data)))
#     # joined_data = [" ".join(d) for d in data]
#     for i in range(len(data)):
#         check_print(i,step=100)
#         for j in range(i,len(data)):
#             # sim = model.wmdistance(joined_data[i], joined_data[j])
#             sim = model.wmdistance(data[i], data[j])
#             sims_df.iloc[i][j] = sims_df.iloc[j][i] = sim
#     sims = np.array(sims_df).tolist()
#     sims = 1 - sims
#     sims,sims_df = normalize(sims,sims_df)
#     return sims,sims_df






# if __name__=="__main__":
#
