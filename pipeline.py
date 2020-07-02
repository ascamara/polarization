import cleaner
import tfidf
import embed
import align
import plot
import nearestk
from matplotlib import pyplot
from collections import OrderedDict
import operator
import numpy as np
import pandas as pd
from time import time


def preprocessing(directory_stem, list_of_corpora, perform=False):
    list_of_corpora_clean = []
    if perform:
        for corpus in list_of_corpora:
            directory = directory_stem + '\\' + corpus
            list_of_corpora_clean.append(cleaner.clean_folder(directory_stem, directory, corpus))
            # to do - get the files into a folder
    # use pre-defined material
    else:
        for corpus in list_of_corpora:
            list_of_corpora_clean.append('clean_{}'.format(corpus))
    return list_of_corpora_clean


def find_terms(directory_stem, list_of_corpora_clean, tfidf_file='tfidf_final', perform=False):
    list_of_terms = []
    if perform:
        list_of_terms = tfidf.tfidf_corpora(directory_stem, list_of_corpora_clean)
    # use pre-defined material
    else:
        list_of_terms = tfidf.read_tfidf(tfidf_file)
    return list_of_terms


def create_models(directory_stem, list_of_corpora_clean, pretrain=False):
    dictionary_of_sentences = embed.create_dictionary_of_sentences(directory_stem, list_of_corpora_clean)
    return embed.create_models(dictionary_of_sentences, pretrain)


def create_co_occurence(directory_stem, dictionary_of_models, list_of_terms):
    dictionary_of_co_occurence = {}
    terms = [terms[0] for terms in list_of_terms[:len(list_of_terms) // 10]]
    for key, value in dictionary_of_models.items():
        stem = directory_stem + '\corpora_{}.csv'
        corpus = key
        print("Working on matrix for: " + key)
        df = pd.read_csv(stem.format(corpus), sep=',')
        df = df.dropna().reset_index(drop=True)
        corpus_arr = [sentences.split() for sentences in df['lines']]
        corpus_set = set()
        for sentence in corpus_arr:
            corpus_set.update(sentence)
        occurrences = OrderedDict((term, OrderedDict((term, 0) for term in corpus_set)) for term in corpus_set)
        for sentence in corpus_arr:
            for i in range(len(sentence)):
                for item in sentence[:i] + sentence[i + 1:]:
                    # if item in terms and sentence[i] in terms:
                    occurrences[sentence[i]][item] += 1
        dictionary_of_co_occurence[key] = occurrences
        with open('matrix_{}'.format(key), 'w', encoding='utf-8') as fp:
            print(occurrences, sep='\n', file=fp)
    return dictionary_of_co_occurence


def fix_model_dimensions(dictionary_of_models):
    base_embed = 0
    for key, value in dictionary_of_models.items():
        if base_embed == 0:
            base_embed = value
        else:
            dictionary_of_models[key] = align.smart_procrustes_align_gensim(base_embed, value)
    return dictionary_of_models


def plot_word(dictionary_of_models, word, k):
    list_of_models = list(dictionary_of_models.values())

    model_1_similarity = list_of_models[0].wv.most_similar(positive=[word], topn=k)
    model_1_comp_dict = [term[0] for term in model_1_similarity]
    model_1_words = [word] + model_1_comp_dict

    model_2_similarity = list_of_models[1].wv.most_similar(positive=[word], topn=k)
    model_2_comp_dict = [term[0] for term in model_2_similarity]
    model_2_words = [word] + model_2_comp_dict
    plot.PCA_plot_diff(list_of_models[0], list_of_models[1], model_1_words, model_2_words, title=word)


def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


if __name__ == '__main__':
    print("Polarization Pipeline V1.0")

    directory_stem = r'C:\Users\ascam\PycharmProjects\polarizat'
    list_of_corpora = ['fox', 'nbc']

    # PART 1: PREPROCESSING
    perform_preprocessing = False
    list_of_corpora_clean = preprocessing(directory_stem, list_of_corpora, perform_preprocessing)

    # PART 2: TF-IDF on the SUBCORPUSES
    perform_tfidf = False
    list_of_terms = find_terms(directory_stem, list_of_corpora_clean, tfidf_file='tfidf_final', perform=False)
    words = [term[0] for term in list_of_terms]

    # PART 3: CREATE and TRAIN MODELS
    # Dictionary of Models: Corp
    dictionary_of_models = create_models(directory_stem, list_of_corpora_clean, pretrain=False)
    # dictionary_of_models = fix_model_dimensions(dictionary_of_models)

    # PART 4: CREATE CO-OCCURENCE MATRIX FOR EACH DICTIONARY
    models_excl_base = without_keys(dictionary_of_models, 'base')
    dictionary_of_co_occurence = create_co_occurence(directory_stem, models_excl_base, list_of_terms)

    # Get (cos sim, importance) then label
    # title = 'Fox News and NBC News Terms from Late March to Early June 2020'
    # plot.cos_sim_plot_2_models(dictionary_of_models, list_of_terms, title)
    list_of_models = list(dictionary_of_models.values())
    list_of_matrices = list(dictionary_of_co_occurence.values())

    nearestk.driver(dictionary_of_models, dictionary_of_co_occurence, list_of_terms, 25, False, True)
    nearestk.driver(dictionary_of_models, dictionary_of_co_occurence, list_of_terms, 25, False, False)
    plot.tight_disp_scatter(dictionary_of_models, list_of_terms)

    # plot_word(dictionary_of_models, word='black', k=25)
    # plot_word(dictionary_of_models, word='mask', k=25)

    '''
    # Find product of TF-IDF and Cosine 'dis-simularity' (1-x)
    products = []
    list_of_models = list(dictionary_of_models.values())
    list_of_terms = [term for term in list_of_terms if
                     term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    for term in list_of_terms:
        vec1, vec2 = list_of_models[0].wv[term[0]], list_of_models[1].wv[term[0]]
        cosine_dissimular = 1 - plot.cosine_sim(vec1, vec2)
        product = term[1] * cosine_dissimular
        products.append((term[0], product))
    products = sorted(products, key=lambda x: x[1])
    products = products[::-1]
    with open('tdidf-sim-products-pretrain', 'w', encoding='utf-8') as fp:
        print(*products, sep='\n', file=fp)
    '''
