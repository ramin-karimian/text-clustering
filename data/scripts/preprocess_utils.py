from multiprocessing import Process, Manager
from data.scripts.preprocess_techs import *
import numpy as np

def preprcess_func(data,numbers,return_dict,prccName):
    total_tokens=[]
    for x in numbers:
        text=data[x]
        if x%1000==0:
            print(f"for {prccName} : {x}")

        # textID = str(x+1)
        text = removeUnicode(text)
        text = replaceURL(text) # Technique 1
        text= replaceEmail(text)
        text = removeHashtagInFrontOfWord(text) # Technique 1 # I
        text = replaceSlang(text) # Technique 2: replaces slang words and abbreviations with their equivalents
        text = replaceContraction(text) # Technique 3: replaces contractions to their equivalents

        # text = replaceAtUser(text) # Technique 1
        # text = removeNumbers(text) # Technique 4: remove integers from text
        # text = removeEmoticons(text) # removes emoticons from text # I attention
        # text = replaceMultiExclamationMark(text) # Technique 5: replaces repetitions of exlamation marks with the tag "multiExclamation"
        # text = replaceMultiQuestionMark(text) # Technique 5: replaces repetitions of question marks with the tag "multiQuestion"
        # text = replaceMultiStopMark(text) # Technique 5: replaces repetitions of stop marks with the tag "multiStop"
        # tokens, err = replaceProperNoun(tokens,textID)

        tokens = tokenize(text)

        total_tokens.append((x,tokens))
        return_dict[x]=tokens
        # return_dict[f"err{prccName[-1]}"]=err

def preprocess_multi_process(func,cores,data):
    processes=[]
    manager= Manager()
    return_dict = manager.dict()

    lenData=len(data)
    t=int(np.floor(lenData/cores))

    for n in range(cores):
        prccName=f"process number = {n+1}"
        print(prccName)
        if n== cores-1:
            numbers=range(n*t,lenData)
        else:
            numbers=range(n*t,(n+1)*t)
        d={}
        for i in numbers:
            # d[i]=data["commentBody"][i]
            d[i]=data["text"][i]
            # d[i]=data["Comment"][i]

        p=Process(target=func,args=(d,numbers,return_dict,prccName))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    data["tokens"]=None
    errs=[]
    for item in return_dict:
        data["tokens"][item]=return_dict[item]

    return data
