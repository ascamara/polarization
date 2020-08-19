import pickle
import math
import cleaner
import tfidf
import embed
import cooccurance
import nearestk
import numpy as np
import logodds
import corpus_measurements
from collections import defaultdict
from operator import itemgetter
from time import time
from gensim.models import KeyedVectors

# Polarization Pipeline

# folder location of the data
directory_stem = r'C:\Users\ascam\PycharmProjects\polarization_pipeline'
# sources to be considered
sources = ['breitbart', 'foxnews', 'reuters', 'cnn', 'huffpo']
# if this is set to false, it uses log odds instead -- this is not ready yet
USE_TFIDF = False
# size of context for controversy scoring
CONTEXT_SIZE = 50
# size of terms considered for corpus-wide measurements
CORPUS_WIDE_MEASUREMENT_TERMS = 10

# try context size: 5, 10, 50
# num terms: 10, 20, 50

## PLEASE SEE THE MAIN FUNCTION TO RUN THE PIPELINE ##


def clean(directory_stem, sources):
    return cleaner.clean_corpus(directory_stem, sources)


def significance(source_dictionary):
    if USE_TFIDF:
        significance_dictionary = tfidf.tfidf_corpus(source_dictionary)
    else:
        significance_dictionary = logodds.logodds_corpus(source_dictionary)
    return significance_dictionary


def controversy(directory_stem, source_dictionary, significance_dictionary):
    significance_list = dict_to_list(significance_dictionary)
    model_dictionary = embed.create_models(directory_stem, source_dictionary, pretrain=False)
    co_occurance_matrix_dictionary = cooccurance.find_matrices(source_dictionary)
    return nearestk.controversy_dictionary_use_co_occurance(model_dictionary, significance_list,
                                                            co_occurance_matrix_dictionary, CONTEXT_SIZE)


def agreement(controversy_dictionary):
    agreement_dictionary = defaultdict(float)
    for key, value in controversy_dictionary.items():
        agreement_dictionary[key] = 1 - value
    return agreement_dictionary


def polarization(significance_dictionary, controversy_dictionary):
    polarization_dictionary = defaultdict(float)
    for key, value in significance_dictionary.items():
        polarization_dictionary[key] = value * controversy_dictionary[key]
    return polarization_dictionary


def adj_polarization(significance_dictionary, controversy_dictionary):
    polarization_dictionary = defaultdict(float)
    max_sig = max([term[1] for term in dict_to_list(significance_dictionary)])
    max_con = max([term[1] for term in dict_to_list(controversy_dictionary)])
    for key, value in significance_dictionary.items():
        polarization_dictionary[key] = math.sqrt(value / max_sig) * (math.sqrt(controversy_dictionary[key] / max_con))
    return polarization_dictionary


def consensus(significance_dictionary, controversy_dictionary):
    polarization_dictionary = defaultdict(float)
    for key, value in significance_dictionary.items():
        polarization_dictionary[key] = value * (1 - controversy_dictionary[key])
    return polarization_dictionary


def adj_consensus(significance_dictionary, controversy_dictionary):
    polarization_dictionary = defaultdict(float)
    max_sig = max([term[1] for term in dict_to_list(significance_dictionary)])
    max_con = max([term[1] for term in dict_to_list(controversy_dictionary)])
    for key, value in significance_dictionary.items():
        polarization_dictionary[key] = math.sqrt(value / max_sig) * (
                1 - math.sqrt(controversy_dictionary[key] / max_con))
    return polarization_dictionary


def score_corpus(significance_dictionary, controversy_dictionary):
    terms, terms_adj = [], []
    weight, weight_adj = [], []
    max_sig = max([term[1] for term in dict_to_list(significance_dictionary)])
    max_con = max([term[1] for term in dict_to_list(controversy_dictionary)])
    for key, value in significance_dictionary.items():
        if controversy_dictionary[key] > 0:
            terms.append(controversy_dictionary[key])
            terms_adj.append(math.sqrt(controversy_dictionary[key] / max_con))
            weight.append(value)
            weight_adj.append(math.sqrt(value / max_sig))
    weighted_avg = np.average(terms, weights=weight)
    weighted_avg_adj = np.average(terms_adj, weights=weight_adj)
    return [weighted_avg, weighted_avg_adj]


def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list


def list_to_txt(sources, list_, txt_name="undef"):
    with open('{}_{}_{}'.format(txt_name, sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print(*list_, sep='\n', file=fp)


def create_report(sources,
                  avg_pol,
                  polarization_dictionary,
                  controversy_dictionary,
                  significance_dictionary):
    sources = [source for source in sources]
    with open('summary_{}_{}'.format(sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print('Polarization pipeline summary for ' + sources[0] + ' and ' + sources[1] + ':', file=fp)
        print('Polarization score: ' + str(avg_pol), file=fp)
        models, matrices = [], []
        for source in sources:
            models.append(KeyedVectors.load_word2vec_format('model_{}.txt'.format(source), binary=False))
            matrices.append(pickle.load(open('matrix_{}'.format(source), 'rb')))
        polarization_list = dict_to_list(polarization_dictionary)[:25]
        controversy_list = dict_to_list(controversy_dictionary)[:25]
        significance_list = dict_to_list(significance_dictionary)[:25]
        print('--- Polarization List ---', file=fp)
        for term in polarization_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)
        print('--- Controversy List ---', file=fp)
        for term in controversy_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)
        print('--- Significance List ---', file=fp)
        for term in significance_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)


def fixlen(significance_dictionary, controversy_dictionary):
    n = len(dict_to_list(controversy_dictionary))
    significance_list = dict_to_list(significance_dictionary)[:n]
    adj_significance_dictionary = {}
    for element in significance_list:
        adj_significance_dictionary[element[0]] = element[1]
    return adj_significance_dictionary


def pipeline(sources):
    # Cleaning
    source_dictionary = clean(directory_stem, sources)

    # Significance scoring
    significance_dictionary = significance(source_dictionary)

    # Controversy scoring
    controversy_dictionary = controversy(directory_stem, source_dictionary, significance_dictionary)
    significance_dictionary = fixlen(significance_dictionary, controversy_dictionary)

    # Polarization scoring
    polarization_dictionary = polarization(significance_dictionary, controversy_dictionary)

    # r^2 of all terms
    r2_all = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                      polarization_dictionary, len(polarization_dictionary))
    # r^2 of top x polariz
    r2_top_polar = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 polarization_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS)
    # r^2 of top X controv
    r2_top_contr = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 controversy_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS)
    # r^2 of top X signif
    r2_top_signf = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 significance_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS)

    create_report(sources,
                  r2_top_contr,
                  polarization_dictionary,
                  controversy_dictionary,
                  significance_dictionary)

    return [r2_all, r2_top_polar, r2_top_contr, r2_top_signf]

if __name__ == '__main__':
    t = time()
    a = pipeline([sources[0], sources[1]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    b = pipeline([sources[3], sources[4]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    c = pipeline([sources[0], sources[4]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    d = pipeline([sources[1], sources[3]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    print(a)
    print(b)
    print(c)
    print(d)




