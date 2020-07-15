import cleaner
import tfidf
import embed
import align
import cooccurance
import plot
import nearestk
import polarization_calculator
import collections
from operator import itemgetter
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from statistics import stdev
import numpy as np
import os


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


def find_terms(directory_stem, list_of_corpora_clean, perform=False, tfidf_file='tfidf_final'):
    list_of_terms = []
    if perform:
        list_of_terms = tfidf.tfidf_corpora(directory_stem, list_of_corpora_clean)
    # use pre-defined material
    else:
        list_of_terms = tfidf.read_tfidf(tfidf_file)
    return list_of_terms


def create_models(directory_stem, list_of_corpora_clean, pretrain=True):
    dictionary_of_sentences = embed.create_dictionary_of_sentences(directory_stem, list_of_corpora_clean)
    return embed.create_models(dictionary_of_sentences, pretrain)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def create_models_random(directory_stem, list_of_corpora_clean, key_names, pretrain=True):
    dictionary_of_sentences = embed.create_dictionary_of_sentences(directory_stem, list_of_corpora_clean)
    dictionary_of_sentences_random = collections.defaultdict(list)
    list_a, list_b = [], []
    for key, value in dictionary_of_sentences.items():
        for group in chunker(dictionary_of_sentences[key], 20):
            # randomize
            val = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if val >= .5:
                for sentence in group:
                    list_a.append(sentence)
            else:
                for sentence in group:
                    list_b.append(sentence)
    dictionary_of_sentences_random[key_names[0]] = embed.fix_arrays(list_a)
    dictionary_of_sentences_random[key_names[1]] = embed.fix_arrays(list_b)
    return embed.create_models(dictionary_of_sentences_random, pretrain)


def read_models(directory_stem, list_of_corpora_clean):
    dictionary_of_models = {}
    for item in list_of_corpora_clean:
        path = 'model_{}.txt'.format(item)
        dictionary_of_models[item] = KeyedVectors.load_word2vec_format(path, binary=False)
    path = 'model_base.txt'
    dictionary_of_models['base'] = KeyedVectors.load_word2vec_format(path, binary=False)
    return dictionary_of_models


def fix_model_dimensions(dictionary_of_models):
    base_embed = 0
    for key, value in dictionary_of_models.items():
        if base_embed == 0:
            base_embed = value
        else:
            dictionary_of_models[key] = align.smart_procrustes_align_gensim(base_embed, value)
    return dictionary_of_models


def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list


if __name__ == '__main__':
    print("Polarization Pipeline")

    directory_stem = r'C:\Users\ascam\PycharmProjects\polarizat'
    list_of_corpora = ['breitbart', 'cnn', 'foxnews', 'huffpo', 'reuters']

    # PART 1: PREPROCESSING
    perform_preprocessing = True
    list_of_corpora_clean = preprocessing(directory_stem,
                                          list_of_corpora,
                                          perform_preprocessing)
    assert False
    # PART 2: TF-IDF on the SUBCORPUSES
    perform_tfidf = True
    dict_of_terms = find_terms(directory_stem,
                               list_of_corpora_clean,
                               perform_tfidf,
                               tfidf_file='tfidf_final')
    list_of_terms = dict_to_list(dict_of_terms)

    # PART 3: CREATE and TRAIN MODELS
    pretrain = True
    generate_new_models = False
    if generate_new_models:
        dictionary_of_models = create_models(directory_stem,
                                             list_of_corpora_clean,
                                             pretrain)
    else:
        dictionary_of_models = read_models(directory_stem,
                                           list_of_corpora_clean)

    # PART 4 (Optional): CREATE CO-OCCURENCE MATRIX FOR EACH DICTIONARY
    models_excl_base = without_keys(dictionary_of_models, 'base')
    perform_cocurrence = False
    perform_matrix_generation = False
    if perform_cocurrence and perform_matrix_generation:
        dictionary_of_co_occurence = cooccurance.create_co_occurence(directory_stem,
                                                                     models_excl_base,
                                                                     list_of_terms)
    elif perform_cocurrence and not perform_matrix_generation:
        dictionary_of_co_occurence = cooccurance.read_co_occurance(models_excl_base)

    # PART 5: FIND R SQUARED
    base_controversy_dictionary = polarization_calculator.controversy_dictionary(dictionary_of_models,
                                                                                 list_of_terms,
                                                                                 k=50,
                                                                                 use_pretrained=False)

    # PART 6: RUN RANDOMIZED PIPELINES
    pretrain = True
    generate_new_models = True
    list_of_randomized_controversy_dictionaries = []
    for i in range(25):
        key_names = ['a_{}'.format(i), 'b_{}'.format(i)]
        if generate_new_models:
            dictionary_of_models = create_models_random(directory_stem,
                                                        list_of_corpora_clean,
                                                        key_names,
                                                        pretrain)
        else:
            dictionary_of_models = read_models(directory_stem,
                                               key_names)
        list_of_randomized_controversy_dictionaries.append(
            polarization_calculator.controversy_dictionary(dictionary_of_models,
                                                           list_of_terms,
                                                           k=50,
                                                           use_pretrained=False))
    stat_significant = {}
    stat_insignificant = {}
    count_of_sig_words = 0
    count_of_all_words = 0
    for key, value in base_controversy_dictionary.items():
        count_of_randoms_greater_than_base = 0
        rand_vals = []
        for random_controversy_dictionary in list_of_randomized_controversy_dictionaries:
            rand_vals.append(random_controversy_dictionary[key])
            if random_controversy_dictionary[key] > base_controversy_dictionary[key]:
                count_of_randoms_greater_than_base += 1
        count_of_all_words += 1
        if count_of_randoms_greater_than_base < 2:
            count_of_sig_words += 1
            stat_significant[key] = (base_controversy_dictionary[key] - (sum(rand_vals) / (len(rand_vals)))) / \
                                    stdev(rand_vals + base_controversy_dictionary[key])
        else:
            stat_insignificant[key] = (base_controversy_dictionary[key] - (sum(rand_vals) / (len(rand_vals)))) / \
                                      stdev(rand_vals + base_controversy_dictionary[key])

    stat_sig_list = dict_to_list(stat_significant)
    with open('stat_sig', 'w', encoding='utf-8') as fp:
        print(*stat_sig_list, sep='\n', file=fp)
    stat_insig_list = dict_to_list(stat_insignificant)
    with open('stat_insig', 'w', encoding='utf-8') as fp:
        print(*stat_insig_list, sep='\n', file=fp)

    # adjustments for significance
    polarization = {}
    polarization_adj = {}
    polar_terms, polar_terms_adj = [], []
    weight, weight_adj = [], []
    max_sig = max([term[1] for term in dict_to_list(dict_of_terms)])
    max_pol = max([term[1] for term in dict_to_list(stat_significant)])
    for key, value in stat_significant.items():
        polarization[key] = stat_significant[key] * dict_of_terms[key]
        polarization_adj[key] = (stat_significant[key] / max_pol) * (dict_of_terms[key] / max_sig)
        polar_terms.append(stat_significant[key])
        polar_terms_adj.append(stat_significant[key] / max_pol)
        weight.append(dict_of_terms[key])
        weight_adj.append(dict_of_terms[key] / max_sig)

    # Print adjusted and weighted scores
    polar = dict_to_list(polarization)
    with open('polar', 'w', encoding='utf-8') as fp:
        print(*polar, sep='\n', file=fp)
    polar_adj = dict_to_list(polarization_adj)
    with open('polar_adj', 'w', encoding='utf-8') as fp:
        print(*polar_adj, sep='\n', file=fp)

    weighted_avg = np.average(polar_terms, weights=weight)
    weighted_avg_adj = np.average(polar_terms_adj, weights=weight_adj)
    print("Polarization score: " + weighted_avg)
    print("Adjusted polarization score: " + weighted_avg_adj)