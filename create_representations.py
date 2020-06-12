from utils_functions.ELMo_encoder import create_elmo_rep
from utils_functions.BERT_encoder import create_bert_rep
from utils_functions.topic_modeling_encoder import create_topical_rep
from utils_functions.universal_sentence_encoder import create_use_rep
from utils_functions.tf_idf_encoder import create_tftfidf_rep
from utils_functions.terms_encoder import create_termstf_rep



if __name__=="__main__":
    # dataset = "bbcsport"
    # dataset = "cran-cisi-pubmed"
    dataset = "cran-cisi-pubmed-overlapped"
    # dataset = "cran-cisi-pubmed-1000"
    # dataset = "cran-cisi-pubmed-1000-overlapped"
    # dataset = "cran-cisi-pubmed-300"
    # dataset = "cran-cisi-pubmed-300-overlapped"
    # dataset = "cran-cisi-pubmed-100"
    # dataset = "cran-cisi-pubmed-100-overlapped"
    # dataset = "mixed"
    # dataset = "cran_cisi"
    # dataset = "pubmed_cisi"
    # dataset = "cran_pubmed"
    # dataset = "twitter"
    # dataset = "mixedOverlap"
    # dataset = "twitter-cor2"
    # data_version="V01"
    data_version="V02"
    # models = [create_bert_rep,create_use_rep,create_elmo_rep,
    #           create_tftfidf_rep,create_termstf_rep,create_topical_rep ]
    # models = [create_elmo_rep,create_topical_rep ]
    models = [create_topical_rep ]

    for model in models:
        print(model)
        model(dataset,data_version)
