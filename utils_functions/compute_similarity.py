import os
from sklearn.metrics.pairwise import cosine_similarity
from utils_functions.utils import *

# def similarity(data):
#     if len(data)>10000:
#         sims = []
#         step = 1
#         # for i in range(step,len(data),step):
#         for i in range(len(data)):
#             if i%1000 ==0 : print(i)
#             # s = cosine_similarity([x for x in data[i-step:i]],[x for x in data])
#             s = cosine_similarity([data[i]],[x for x in data])
#             sims.extend(np.around(s.tolist(),2))
#         if i != len(data):
#             s = cosine_similarity([x for x in data[i:len(data)]],[x for x in data])
#             sims.extend(np.around(s.tolist(),2))
#         sims = np.around(sims,2)
#     else:
#         sims = cosine_similarity([x for x in data],[x for x in data])
#         sims = np.around(sims.tolist(),2)
#     sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
#     return sims_df,sims

def similarity(data):
    sims = cosine_similarity([x for x in data],[x for x in data])
    sims = np.around(sims.tolist(),2)
    sims_df = pd.DataFrame(sims,index = range(len(sims)), columns= range(len(sims)))
    return sims_df,sims

def similarity_1by1(vec1,vec2):
    sim = cosine_similarity([vec1],[vec2])[0][0]
    return sim

if __name__=="__main__":
    datapath="data/preprocessed_data(polarity_added).pkl"
    emb_path= "data/source_data/embeddings_index(from_GoogleNews-vectors-negative300).pkl"
    oneOrTotal = ["one_article","total","total_one_article"][2]
    dirname = f"my_word2vec_model_{oneOrTotal}"
    path = f"models/"+dirname

    data = load_data(datapath,article=oneOrTotal)
    data = data[0]
    artId= data["articleID"][0].split("/")[-1]
    if dirname not in os.listdir("models"):
        os.mkdir(path)
    simspath = path + f"/{dirname}_similarities_({artId}).pkl"
    embpath = path + f"/{dirname}_embeddings_({artId}).pkl"
    dfpath = path + f"/{dirname}.pkl"

    emb , df =embs(data,emb_path)

    sims = similarity(emb)
    sims = pd.DataFrame(sims,index = data["commentID"], columns= data["commentID"])
    sims = pd.DataFrame(sims,index = data["commentID"], columns= data["commentID"])
    save_data(simspath,sims)
    save_data(embpath,emb)
    save_data(dfpath,df)
    sims.to_csv(simspath[:-4]+".csv")
    df.to_csv(dfpath[:-4]+".csv")
