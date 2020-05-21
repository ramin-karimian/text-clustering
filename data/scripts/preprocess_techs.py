from spellchecker import SpellChecker
import re
from functools import partial
from redditscore.tokenizer import CrazyTokenizer
from nltk.corpus import wordnet , names
from nltk.corpus import stopwords
from data.scripts.utils import *
import pandas as pd
from nltk import pos_tag

# embpath =f"G:\projects_backup\pj1-hashtag-recommendation\pickles\embeddings_index(from_GoogleNews-vectors-negative300).pkl"
# emb = load_data(embpath)
# emb = list(emb.keys())

stop_words = stopwords.words('english')
stop_words.extend(['rt'])
[stop_words.remove(x) for x in
                [  'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                   "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                   "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                   'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                   'won', "won't", 'wouldn', "wouldn't", 'can', 'will', 'just', 'don', "don't",
                   'should', "should've", 'than', 'too', 'very',
                   'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                   'no', 'nor', 'not', 'only', 'same', 'so' ,  'against', 'between',
                   'into', 'through', 'during', 'before', 'after', 'above', 'below',
                   'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'but',
                    'do', 'does', 'did', 'doing'
                   ] ]

# s1 = stopwords.words('english')
# s2 = pd.read_excel("source_data/Top 300 words.xlsx",sheet_name="stop words",header=None)[0]
# s2 = list(s2.values)
# for x in s1:
#     if x not in s2:
#         s2.append(x)
# addList = ["'s", 'nt', 'ms.', "n't", 'al', 'oh', 'uma', 'nyt', "'re", '1', '2', '20', 'ca', '3', '10', 'and/or', '50', "nyong'o", 'ed', '100', 'url', '30', '40', 'ny', '5', 'dr.', '4', '60', "'ve", '0', '15', 0, '25', '70', 12, '6', 'et', '+', "'l", '16', 'sen.', '14', '8', '2016', '13', 'pr', '80', '2020', '90', '7', '11', '9', '22', "'d", '19', '17', 'ah', '99', '35', 'na', 'j.', '2015', '~', 'se', '45', '1970', '95', 'uh', 'mrs.', 'hi', '@mor', 'b.', 'san', '24', 'ix', 'an', '65', '1st', 'ps', 'ta', '21', '23', '58', "'em", 'yi', '26', '201', '@mark', 'er', '27', 'ha', '52', 'en', '75', 'w.', 'jr.', '2014', 'dr', 'p.s.', '500', '2019', '@mulder', '1960', '85', '@dr', 'c.k.', '47', '190', 'nyts', '195', '1000', '24/7', '63', '55', 'lol', 'i.', '68', 'ho', 'h.', '51', 'yo', '1971', '36', '197', 'ii', '41', 'r.', '1950', 'layer', '48', '@jef', 'eh', '@john', '64', '191', 'snl', 'c.', 'mis', '40s', 'yr', '300', '@mike', 'mph', '@bil', '@ludwig', 'nah', 'd.c.', 'k.', '28', 'hs', '1984', '202', '61', 'bt', '203', '5th', '@robert', '@jonathan', '72', 'v.', 'las', 'co.', '74', 'pal', '1975', 'm.', 's.', '194', 'ct', '2010', '193', 'l.', '71', '62', '@tom', '31', '@from']
# for x in addList:
#     if x not in s2:
#         s2.append(x)
# removeList = ['should','could','due']
# [s2.remove(x) for x in removeList if x in s2 ]
# # [s2.remove(x) for x in [  'should','could','due' ] if x in s2 ]
# stop_words = s2

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    text = re.sub(r'(\\x[a-zA-Z0-9]{2})',r'', text)
    text = re.sub(r'(\\[nr]+)',r' ', text)
    return text

def replaceAtUser(text):
    """ Replaces "@user" with "username" """
#    text = re.sub('@[^\s]+','atUser',text)
    text = re.sub('@[^\s]+','',text)
    return text

def replaceURL(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)  # url deleted
    return text

def replaceEmail(text):
    """ Replaces url address with "url" """
    text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b','email',text)  # url deleted
    return text

def removeHashtagInFrontOfWord(text):
    """ Removes hashtag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def removeHashtagedWords(text):
    """ Removes hashtaged words in text """
    text = re.sub(r'#([^\s]+)', '', text)
    return text

def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return replaceElongated(repl_word)
    else:
        return repl_word

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def spellCorection(word):
    spell = SpellChecker()
    word = spell.correction(word)
    return word

def remove_pos_taged_tokens(tokens,remove_pos_tags):
    taged_tokens = pos_tag(tokens)
    for i in range(len(taged_tokens)-1,-1,-1):
        if taged_tokens[i][1] in remove_pos_tags:
            del tokens[i]
    return tokens

def tokenize(text,remove_pos_tags,extend_stoplist):
    tokenizer = CrazyTokenizer()
    tokens = tokenizer.tokenize(text)

    if len(remove_pos_tags) > 0 :
        tokens = remove_pos_taged_tokens(tokens,remove_pos_tags)

    preprocessed_tokens = []
    for w in tokens:
        w = replaceElongated(w)
        # if w not in stop_words:
        if w not in stop_words and w not in extend_stoplist:
            # if w in emb:
            # w = spellCorection(w)
            preprocessed_tokens.append(w)
    return preprocessed_tokens


""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True) # longest first for regex
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])



""" Replaces contractions from a string to their equivalents """

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replaceContraction(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)
